const express = require("express");
const { z } = require("zod");
const { asyncHandler } = require("../lib/async-handler");
const { authenticate, requireVerifiedUser } = require("../middleware/auth.middleware");
const {
  authLimiter,
  emailActionLimiter,
  loginLimiter,
  signupLimiter,
  passwordResetLimiter,
} = require("../middleware/rate-limiters");
const {
  signUp,
  verifyEmail,
  resendVerificationEmail,
  login,
  forgotPassword,
  resetPassword,
  getCurrentUser,
} = require("../services/auth.service");
const { getClientIp, getUserAgent } = require("../utils/request-metadata");

const router = express.Router();

router.post(
  "/signup",
  signupLimiter,
  asyncHandler(async (req, res) => {
    const result = await signUp(req.body);
    res.status(201).json(result);
  })
);

router.get(
  "/verify-email",
  authLimiter,
  asyncHandler(async (req, res) => {
    const query = z.object({ token: z.string().min(20) }).parse(req.query);
    const result = await verifyEmail(query.token);
    res.json(result);
  })
);

router.post(
  "/resend-verification",
  emailActionLimiter,
  asyncHandler(async (req, res) => {
    const body = z.object({ email: z.string().trim().email() }).parse(req.body);
    const result = await resendVerificationEmail(body.email);
    res.json(result);
  })
);

router.post(
  "/login",
  loginLimiter,
  asyncHandler(async (req, res) => {
    const result = await login(req.body, {
      ipAddress: getClientIp(req),
      userAgent: getUserAgent(req),
    });
    res.json(result);
  })
);

router.post(
  "/forgot-password",
  emailActionLimiter,
  asyncHandler(async (req, res) => {
    const result = await forgotPassword(req.body);
    res.json(result);
  })
);

router.post(
  "/reset-password",
  passwordResetLimiter,
  asyncHandler(async (req, res) => {
    const result = await resetPassword(req.body);
    res.json(result);
  })
);

router.get(
  "/me",
  authenticate,
  requireVerifiedUser,
  asyncHandler(async (req, res) => {
    const result = await getCurrentUser(req.auth.sub);
    res.json(result);
  })
);

module.exports = { authRouter: router };
