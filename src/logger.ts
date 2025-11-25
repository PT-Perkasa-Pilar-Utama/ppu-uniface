/**
 * Configuration for the logging system
 */
export class LoggerConfig {
  /** Enable or disable verbose logging */
  static verbose: boolean = false;
}

/**
 * Logs a message with timestamp, class name, and method name
 * @param className - Name of the class
 * @param methodName - Name of the method
 * @param message - Log message
 */
export function logger(
  className: string,
  methodName: string,
  message: string
): void {
  if (!LoggerConfig.verbose) return;

  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${className}.${methodName}: ${message}`);
}
