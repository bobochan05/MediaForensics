const rateLimit = require("express-rate-limit");

function jsonRateLimit(options) {
  return rateLimit({
    standardHeaders: true,
    legacyHeaders: false,
    handler: (_req, res) => {
      res.status(429).json({
        error: options.message || "Too many requests. Please try again later.",
      });
    },
    ...options,
  });
}

function keyByIpAndEmail(req) {
  const email = String(req.body?.email || "").trim().toLowerCase();
  return `${req.ip}:${email || "anonymous"}`;
}

const authLimiter = jsonRateLimit({
  windowMs: 15 * 60 * 1000,
  max: 60,
  message: "Too many authentication requests. Please try again later.",
});

const loginLimiter = jsonRateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: "Too many login attempts. Please wait before retrying.",
});

const signupLimiter = jsonRateLimit({
  windowMs: 60 * 60 * 1000,
  max: 10,
  message: "Too many signup attempts. Please try again later.",
});

const passwordResetLimiter = jsonRateLimit({
  windowMs: 60 * 60 * 1000,
  max: 10,
  message: "Too many password reset requests. Please try again later.",
});

const emailActionLimiter = jsonRateLimit({
  windowMs: 60 * 60 * 1000,
  max: 5,
  keyGenerator: keyByIpAndEmail,
  message: "Too many email requests for this account. Please try again later.",
});

module.exports = {
  authLimiter,
  loginLimiter,
  signupLimiter,
  passwordResetLimiter,
  emailActionLimiter,
};
