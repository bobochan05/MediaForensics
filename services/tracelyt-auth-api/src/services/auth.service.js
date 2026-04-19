const jwt = require("jsonwebtoken");
const { EmailTokenType, UserStatus } = require("@prisma/client");
const { z } = require("zod");
const { env } = require("../config/env");
const { AppError } = require("../errors/app-error");
const { prisma } = require("../lib/prisma");
const {
  emailQueue,
  EmailJobType,
  buildDedupeWindow,
} = require("./email/email-queue");
const { buildDeviceProfile } = require("../utils/device-fingerprint");
const { hashPassword, verifyPassword } = require("../utils/security");
const {
  assertEmailCooldown,
  issueEmailToken,
  consumeEmailToken,
  buildVerificationUrl,
  buildPasswordResetUrl,
} = require("./token.service");

const signupSchema = z.object({
  email: z.string().trim().email(),
  fullName: z.string().trim().min(2).max(120),
  password: z.string().min(10).max(128),
});

const loginSchema = z.object({
  email: z.string().trim().email(),
  password: z.string().min(1).max(128),
});

const forgotPasswordSchema = z.object({
  email: z.string().trim().email(),
});

const resetPasswordSchema = z.object({
  token: z.string().min(20),
  password: z.string().min(10).max(128),
});

function createAccessToken(user) {
  return jwt.sign(
    {
      sub: user.id,
      email: user.email,
      fullName: user.fullName,
      emailVerified: user.emailVerified,
      status: user.status,
    },
    env.jwtSecret,
    { expiresIn: env.jwtExpiresIn }
  );
}

async function queueVerificationEmail(user, db = prisma) {
  await assertEmailCooldown({
    userId: user.id,
    type: EmailTokenType.VERIFY_EMAIL,
    cooldownSeconds: env.verificationEmailCooldownSeconds,
    db,
  });

  const { rawToken, tokenRecord } = await issueEmailToken({
    userId: user.id,
    type: EmailTokenType.VERIFY_EMAIL,
    ttlMinutes: env.emailVerificationTtlMinutes,
    db,
  });

  await emailQueue.enqueue(
    {
      type: EmailJobType.VERIFY_EMAIL,
      userId: user.id,
      tokenId: tokenRecord.id,
      payload: {
        email: user.email,
        fullName: user.fullName,
        verificationUrl: buildVerificationUrl(rawToken),
        expiresInMinutes: env.emailVerificationTtlMinutes,
      },
    },
    db
  );
}

async function queuePasswordResetEmail(user, db = prisma) {
  await assertEmailCooldown({
    userId: user.id,
    type: EmailTokenType.PASSWORD_RESET,
    cooldownSeconds: env.passwordResetCooldownSeconds,
    db,
  });

  const { rawToken, tokenRecord } = await issueEmailToken({
    userId: user.id,
    type: EmailTokenType.PASSWORD_RESET,
    ttlMinutes: env.passwordResetTtlMinutes,
    db,
  });

  await emailQueue.enqueue(
    {
      type: EmailJobType.PASSWORD_RESET,
      userId: user.id,
      tokenId: tokenRecord.id,
      payload: {
        email: user.email,
        fullName: user.fullName,
        resetUrl: buildPasswordResetUrl(rawToken),
        expiresInMinutes: env.passwordResetTtlMinutes,
      },
    },
    db
  );
}

async function signUp(payload) {
  const parsed = signupSchema.parse(payload);
  const email = parsed.email.toLowerCase();

  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) {
    throw new AppError(409, "An account with this email already exists.");
  }
  const passwordHash = await hashPassword(parsed.password);
  const user = await prisma.$transaction(async (tx) => {
    const createdUser = await tx.user.create({
      data: {
        email,
        fullName: parsed.fullName,
        passwordHash,
        emailVerified: false,
        status: UserStatus.PENDING,
      },
    });

    await queueVerificationEmail(createdUser, tx);
    return createdUser;
  });

  return {
    message: "Account created. Verify your email to activate sign-in.",
    user: {
      id: user.id,
      email: user.email,
      fullName: user.fullName,
      emailVerified: user.emailVerified,
      status: user.status,
    },
  };
}

async function verifyEmail(token) {
  const result = await consumeEmailToken({
    rawToken: token,
    type: EmailTokenType.VERIFY_EMAIL,
  });

  await prisma.user.update({
    where: { id: result.userId },
    data: {
      emailVerified: true,
      verifiedAt: new Date(),
      status: UserStatus.ACTIVE,
    },
  });

  return {
    message: "Email verified successfully. Your account is now active.",
  };
}

async function resendVerificationEmail(email) {
  const normalized = String(email || "").trim().toLowerCase();
  const user = await prisma.user.findUnique({ where: { email: normalized } });

  if (!user || user.emailVerified) {
    return {
      message:
        "If the account exists and still needs verification, a verification email will be queued.",
    };
  }

  await prisma.$transaction(async (tx) => {
    await queueVerificationEmail(user, tx);
  });
  return {
    message:
      "If the account exists and still needs verification, a verification email will be queued.",
  };
}

function buildLoginAlertDedupeKey(userId, deviceFingerprint, ipAddress) {
  return [
    "new-login-alert",
    userId,
    deviceFingerprint,
    ipAddress,
    buildDedupeWindow(env.loginAlertCooldownHours),
  ].join(":");
}

async function createLoginEventAndAlerts(user, { ipAddress, userAgent }) {
  const device = buildDeviceProfile(userAgent);
  const [knownDevice, knownIp] = await Promise.all([
    prisma.loginEvent.findFirst({
      where: {
        userId: user.id,
        deviceFingerprint: device.fingerprint,
      },
    }),
    prisma.loginEvent.findFirst({
      where: {
        userId: user.id,
        ipAddress,
      },
    }),
  ]);

  const isNewDevice = !knownDevice;
  const isNewIp = !knownIp;
  let alertSent = false;

  if (isNewDevice || isNewIp) {
    const dedupeKey = buildLoginAlertDedupeKey(
      user.id,
      device.fingerprint,
      ipAddress
    );

    const queuedAlert = await emailQueue.enqueue({
      type: EmailJobType.NEW_LOGIN_ALERT,
      userId: user.id,
      dedupeKey,
      payload: {
        email: user.email,
        fullName: user.fullName,
        ipAddress,
        deviceLabel: device.label,
        timestamp: new Date().toISOString(),
      },
    });

    alertSent = queuedAlert.created;
  }

  await prisma.loginEvent.create({
    data: {
      userId: user.id,
      ipAddress,
      userAgent,
      deviceFingerprint: device.fingerprint,
      deviceLabel: device.label,
      isNewDevice,
      isNewIp,
      alertSent,
    },
  });

  await prisma.user.update({
    where: { id: user.id },
    data: { lastLoginAt: new Date() },
  });
}

async function login(payload, requestMetadata) {
  const parsed = loginSchema.parse(payload);
  const email = parsed.email.toLowerCase();
  const user = await prisma.user.findUnique({ where: { email } });

  if (!user) {
    throw new AppError(401, "Invalid email or password.");
  }

  const isValidPassword = await verifyPassword(parsed.password, user.passwordHash);
  if (!isValidPassword) {
    throw new AppError(401, "Invalid email or password.");
  }

  if (!user.emailVerified || user.status === UserStatus.PENDING) {
    throw new AppError(403, "Email verification required before login.");
  }

  if (user.status === UserStatus.DISABLED) {
    throw new AppError(403, "This account is disabled.");
  }

  await createLoginEventAndAlerts(user, requestMetadata);

  return {
    message: "Login successful.",
    accessToken: createAccessToken(user),
    user: {
      id: user.id,
      email: user.email,
      fullName: user.fullName,
      emailVerified: user.emailVerified,
    },
  };
}

async function forgotPassword(payload) {
  const parsed = forgotPasswordSchema.parse(payload);
  const user = await prisma.user.findUnique({
    where: { email: parsed.email.toLowerCase() },
  });

  if (!user) {
    return {
      message: "If the account exists, a password reset email will be queued.",
    };
  }

  await prisma.$transaction(async (tx) => {
    await queuePasswordResetEmail(user, tx);
  });

  return {
    message: "If the account exists, a password reset email will be queued.",
  };
}

async function resetPassword(payload) {
  const parsed = resetPasswordSchema.parse(payload);
  const result = await consumeEmailToken({
    rawToken: parsed.token,
    type: EmailTokenType.PASSWORD_RESET,
  });

  await prisma.$transaction(async (tx) => {
    await tx.user.update({
      where: { id: result.userId },
      data: {
        passwordHash: await hashPassword(parsed.password),
        passwordChangedAt: new Date(),
      },
    });

    await tx.emailToken.updateMany({
      where: {
        userId: result.userId,
        type: EmailTokenType.PASSWORD_RESET,
        usedAt: null,
        invalidatedAt: null,
      },
      data: {
        invalidatedAt: new Date(),
      },
    });
  });

  return {
    message: "Password updated successfully.",
  };
}

async function getCurrentUser(userId) {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      fullName: true,
      emailVerified: true,
      createdAt: true,
      verifiedAt: true,
      lastLoginAt: true,
      status: true,
    },
  });

  if (!user) {
    throw new AppError(404, "User not found.");
  }

  return { user };
}

module.exports = {
  signUp,
  verifyEmail,
  resendVerificationEmail,
  login,
  forgotPassword,
  resetPassword,
  getCurrentUser,
};
