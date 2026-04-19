const path = require("path");
const dotenv = require("dotenv");

dotenv.config({
  path: path.resolve(__dirname, "../../.env"),
});

function required(name) {
  const value = process.env[name];
  if (!value || !String(value).trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return String(value).trim();
}

function optional(name, fallback = "") {
  const value = process.env[name];
  return value == null || String(value).trim() === "" ? fallback : String(value).trim();
}

function integer(name, fallback) {
  const raw = optional(name, String(fallback));
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed)) {
    throw new Error(`Environment variable ${name} must be an integer.`);
  }
  return parsed;
}

const env = {
  nodeEnv: optional("NODE_ENV", "development"),
  port: integer("PORT", 4100),
  appBaseUrl: required("APP_BASE_URL"),
  databaseUrl: required("DATABASE_URL"),
  jwtSecret: required("JWT_SECRET"),
  jwtExpiresIn: optional("JWT_EXPIRES_IN", "15m"),
  emailProvider: optional("EMAIL_PROVIDER", "resend").toLowerCase(),
  emailFrom: required("EMAIL_FROM"),
  resendApiKey: optional("RESEND_API_KEY"),
  sendgridApiKey: optional("SENDGRID_API_KEY"),
  emailVerificationTtlMinutes: integer("EMAIL_VERIFICATION_TTL_MINUTES", 24 * 60),
  passwordResetTtlMinutes: integer("PASSWORD_RESET_TTL_MINUTES", 30),
  verificationEmailCooldownSeconds: integer("VERIFICATION_EMAIL_COOLDOWN_SECONDS", 300),
  passwordResetCooldownSeconds: integer("PASSWORD_RESET_COOLDOWN_SECONDS", 300),
  loginAlertCooldownHours: integer("LOGIN_ALERT_COOLDOWN_HOURS", 24),
  emailQueuePollIntervalMs: integer("EMAIL_QUEUE_POLL_INTERVAL_MS", 1500),
  emailQueueLockTimeoutSeconds: integer("EMAIL_QUEUE_LOCK_TIMEOUT_SECONDS", 60),
  emailQueueMaxAttempts: integer("EMAIL_QUEUE_MAX_ATTEMPTS", 5),
  emailQueueBaseBackoffSeconds: integer("EMAIL_QUEUE_BASE_BACKOFF_SECONDS", 30),
  corsOrigin: optional("CORS_ORIGIN", "http://localhost:3000"),
};

if (env.emailProvider === "resend" && !env.resendApiKey) {
  throw new Error("EMAIL_PROVIDER is 'resend' but RESEND_API_KEY is missing.");
}

if (env.emailProvider === "sendgrid" && !env.sendgridApiKey) {
  throw new Error("EMAIL_PROVIDER is 'sendgrid' but SENDGRID_API_KEY is missing.");
}

module.exports = { env };
