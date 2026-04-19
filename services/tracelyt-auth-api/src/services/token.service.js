const { EmailDeliveryStatus, EmailTokenType } = require("@prisma/client");
const { env } = require("../config/env");
const { AppError } = require("../errors/app-error");
const { prisma } = require("../lib/prisma");
const { createRandomToken, hashToken } = require("../utils/security");

function buildExpiry(minutes) {
  return new Date(Date.now() + minutes * 60 * 1000);
}

async function assertEmailCooldown({ userId, type, cooldownSeconds, db = prisma }) {
  const latest = await db.emailToken.findFirst({
    where: {
      userId,
      type,
      invalidatedAt: null,
      OR: [
        { deliveryStatus: EmailDeliveryStatus.PENDING },
        { deliveryStatus: EmailDeliveryStatus.PROCESSING },
        { deliveryStatus: EmailDeliveryStatus.SENT },
      ],
    },
    orderBy: {
      createdAt: "desc",
    },
  });

  if (!latest) {
    return;
  }

  const lastEventAt =
    latest.sentAt || latest.lastSentAttemptAt || latest.createdAt;
  const elapsedMs = Date.now() - new Date(lastEventAt).getTime();
  if (elapsedMs < cooldownSeconds * 1000) {
    throw new AppError(429, "Please wait before requesting another email.");
  }
}

async function issueEmailToken({
  userId,
  type,
  ttlMinutes,
  contextJson,
  db = prisma,
}) {
  await db.emailToken.updateMany({
    where: {
      userId,
      type,
      usedAt: null,
      invalidatedAt: null,
    },
    data: {
      invalidatedAt: new Date(),
    },
  });

  const rawToken = createRandomToken(32);
  const tokenRecord = await db.emailToken.create({
    data: {
      userId,
      type,
      hashedToken: hashToken(rawToken),
      expiresAt: buildExpiry(ttlMinutes),
      contextJson: contextJson || undefined,
      deliveryStatus: EmailDeliveryStatus.PENDING,
    },
  });

  return {
    rawToken,
    tokenRecord,
  };
}

async function consumeEmailToken({ rawToken, type, db = prisma }) {
  const hashed = hashToken(rawToken);

  return db.$transaction(async (tx) => {
    const token = await tx.emailToken.findFirst({
      where: {
        hashedToken: hashed,
        type,
        usedAt: null,
        invalidatedAt: null,
        expiresAt: {
          gt: new Date(),
        },
      },
      include: {
        user: true,
      },
    });

    if (!token) {
      throw new AppError(400, "Token is invalid or expired.");
    }

    const consumeResult = await tx.emailToken.updateMany({
      where: {
        id: token.id,
        usedAt: null,
        invalidatedAt: null,
        expiresAt: {
          gt: new Date(),
        },
      },
      data: { usedAt: new Date() },
    });

    if (consumeResult.count !== 1) {
      throw new AppError(400, "Token is invalid or expired.");
    }

    return token;
  });
}

async function markEmailTokenAttempt({
  tokenId,
  providerName,
  errorMessage,
  db = prisma,
}) {
  if (!tokenId) {
    return null;
  }

  return db.emailToken.update({
    where: { id: tokenId },
    data: {
      sendAttempts: {
        increment: 1,
      },
      lastSentAttemptAt: new Date(),
      deliveryStatus: EmailDeliveryStatus.PROCESSING,
      providerName: providerName || undefined,
      lastDeliveryError: errorMessage || null,
    },
  });
}

async function markEmailTokenSent({
  tokenId,
  providerName,
  providerMessageId,
  db = prisma,
}) {
  if (!tokenId) {
    return null;
  }

  return db.emailToken.update({
    where: { id: tokenId },
    data: {
      sentAt: new Date(),
      lastSentAttemptAt: new Date(),
      deliveryStatus: EmailDeliveryStatus.SENT,
      providerName: providerName || undefined,
      providerMessageId: providerMessageId || undefined,
      lastDeliveryError: null,
    },
  });
}

async function markEmailTokenPendingRetry({ tokenId, errorMessage, db = prisma }) {
  if (!tokenId) {
    return null;
  }

  return db.emailToken.update({
    where: { id: tokenId },
    data: {
      deliveryStatus: EmailDeliveryStatus.PENDING,
      lastDeliveryError: errorMessage || null,
    },
  });
}

async function markEmailTokenFailed({ tokenId, errorMessage, db = prisma }) {
  if (!tokenId) {
    return null;
  }

  return db.emailToken.update({
    where: { id: tokenId },
    data: {
      deliveryStatus: EmailDeliveryStatus.FAILED,
      lastDeliveryError: errorMessage || null,
    },
  });
}

function buildVerificationUrl(rawToken) {
  return `${env.appBaseUrl}/api/auth/verify-email?token=${encodeURIComponent(rawToken)}`;
}

function buildPasswordResetUrl(rawToken) {
  return `${env.appBaseUrl}/reset-password?token=${encodeURIComponent(rawToken)}`;
}

module.exports = {
  EmailTokenType,
  assertEmailCooldown,
  issueEmailToken,
  consumeEmailToken,
  markEmailTokenAttempt,
  markEmailTokenSent,
  markEmailTokenPendingRetry,
  markEmailTokenFailed,
  buildVerificationUrl,
  buildPasswordResetUrl,
};
