import Foundation
import Logging
import Darwin

/// Centralized loggers for audio model subsystems.
public enum AudioLog {
    /// Logger for model weight loading and initialization.
    public static let modelLoading = Logger(label: "com.qwen3speech.model-loading")
    /// Logger for inference and generation.
    public static let inference = Logger(label: "com.qwen3speech.inference")
    /// Logger for HuggingFace downloads and caching.
    public static let download = Logger(label: "com.qwen3speech.download")
    /// Logger for voice pipeline events.
    public static let pipeline = Logger(label: "com.qwen3speech.pipeline")
}

/// Short thread identifier for the calling thread, e.g. `T:576432`.
///
/// Uses `pthread_threadid_np`, which returns a stable 64-bit system-wide thread ID
/// without consuming a Mach port right. Useful for correlating log lines across
/// concurrent tasks in terminal-visible output.
public var currentThreadTag: String {
    var tid: UInt64 = 0
    pthread_threadid_np(nil, &tid)
    return "T:\(tid)"
}

// MARK: - Logging Bootstrap

/// Configure the global swift-log handler to write to stdout.
///
/// Call this once at process startup, before any `Logger` is first accessed.
/// The log level is read from the `LOG_LEVEL` environment variable
/// (`trace`, `debug`, `info`, `notice`, `warning`, `error`, `critical`).
/// Defaults to `info` when the variable is absent or unrecognised.
///
/// All subsystems — Hummingbird request logs, registry logs, pipeline logs — flow
/// through a single stdout stream after this call, observable with a plain `tail -f`.
public func bootstrapLogging() {
    let levelStr = ProcessInfo.processInfo.environment["LOG_LEVEL"] ?? ""
    let level = Logger.Level(rawValue: levelStr.lowercased()) ?? .info
    LoggingSystem.bootstrap { label in
        var handler = StreamLogHandler.standardOutput(label: label)
        handler.logLevel = level
        return handler
    }
}
