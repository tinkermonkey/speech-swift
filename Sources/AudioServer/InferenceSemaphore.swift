import Foundation

/// A simple async counting semaphore that gates concurrent GPU inference calls.
///
/// MLX inference is not safe to parallelize on shared model instances — two
/// concurrent calls race for the same GPU command queue and both run 4-7× slower
/// than they would alone.  This semaphore serializes (or limits) inference so
/// each call gets the full GPU budget.
///
/// Configured via `--concurrency N` (default 1).  Setting N > 1 requires N sets
/// of model instances loaded in memory (see `--workers` if that is ever added).
actor InferenceSemaphore {
    private var available: Int
    private var waiters: [(id: Int, continuation: CheckedContinuation<Bool, Never>)] = []
    private var nextID = 0

    init(permits: Int) {
        precondition(permits >= 1, "permits must be ≥ 1")
        self.available = permits
    }

    /// Acquire a permit. Returns `true` if the permit was granted, `false` if the
    /// calling task was cancelled while waiting (permit was NOT consumed).
    private func wait() async -> Bool {
        if available > 0 {
            available -= 1
            return true
        }
        let id = nextID
        nextID &+= 1
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                waiters.append((id: id, continuation: continuation))
            }
        } onCancel: {
            // onCancel runs outside the actor; hop back in to clean up.
            Task { await self.cancelWaiter(id: id) }
        }
    }

    /// Called from the cancellation handler when a waiting task is cancelled.
    /// Resumes the continuation with `false` (no permit given) so `withPermit` can
    /// throw `CancellationError` without calling `signal()`.
    private func cancelWaiter(id: Int) {
        guard let idx = waiters.firstIndex(where: { $0.id == id }) else {
            // Already dequeued by signal() — the task will get the permit and discover
            // it's cancelled inside body(); signal() in defer releases it correctly.
            return
        }
        waiters.remove(at: idx).continuation.resume(returning: false)
        // Available is not changed: this waiter never held a permit.
    }

    /// Release a permit, resuming the longest-waiting caller if any.
    func signal() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.continuation.resume(returning: true)  // permit transferred
        } else {
            available += 1
        }
    }

    /// Number of tasks currently waiting to acquire a permit.
    var waitingCount: Int { waiters.count }

    /// Run `body` with a permit held, releasing it when `body` returns or throws.
    /// Throws `CancellationError` if the calling task was cancelled while waiting.
    func withPermit<T: Sendable>(_ body: () async throws -> T) async throws -> T {
        guard await wait() else {
            throw CancellationError()
        }
        defer { signal() }
        return try await body()
    }
}
