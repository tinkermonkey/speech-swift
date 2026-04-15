import os
import Darwin

/// Centralized loggers for audio model subsystems.
public enum AudioLog {
    /// Logger for model weight loading and initialization.
    public static let modelLoading = Logger(subsystem: "com.qwen3speech", category: "ModelLoading")
    /// Logger for inference and generation.
    public static let inference = Logger(subsystem: "com.qwen3speech", category: "Inference")
    /// Logger for HuggingFace downloads and caching.
    public static let download = Logger(subsystem: "com.qwen3speech", category: "Download")
    /// Logger for voice pipeline events.
    public static let pipeline = Logger(subsystem: "com.qwen3speech", category: "Pipeline")
}

/// Short thread identifier for the calling thread, e.g. `T:576432`.
///
/// Uses `pthread_threadid_np`, which returns a stable 64-bit system-wide thread ID
/// without consuming a Mach port right. Useful for correlating log lines across
/// concurrent tasks in terminal-visible output.
/// Note: `os.Logger` already captures thread IDs automatically at the OS level
/// (visible in Console.app). This is for embedding in terminal-visible log output.
public var currentThreadTag: String {
    var tid: UInt64 = 0
    pthread_threadid_np(nil, &tid)
    return "T:\(tid)"
}
