const jwt = require("jsonwebtoken");
const { env } = require("../config/env");
const { AppError } = require("../errors/app-error");

function authenticate(req, _res, next) {
  const raw = req.headers.authorization || "";
  const [scheme, token] = raw.split(" ");

  if (!token || scheme !== "Bearer") {
    return next(new AppError(401, "Authentication required."));
  }

  try {
    req.auth = jwt.verify(token, env.jwtSecret);
    return next();
  } catch (_error) {
    return next(new AppError(401, "Invalid or expired access token."));
  }
}

function requireVerifiedUser(req, _res, next) {
  if (!req.auth) {
    return next(new AppError(401, "Authentication required."));
  }

  if (!req.auth.emailVerified) {
    return next(new AppError(403, "Email verification required."));
  }

  return next();
}

module.exports = {
  authenticate,
  requireVerifiedUser,
};
