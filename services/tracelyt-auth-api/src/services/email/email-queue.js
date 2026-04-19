const {
  EmailDeliveryStatus,
  EmailJobType,
  Prisma,
} = require("@prisma/client");
const { env } = require("../../config/env");
const { prisma } = require("../../lib/prisma");
const {
  markEmailTokenAttempt,
  markEmailTokenFailed,
  markEmailTokenPendingRetry,
  markEmailTokenSent,
} = require("../token.service");
const { EmailService } = require("./email.service");

function truncateError(message) {
  return String(message || "Unknown email queue failure.").slice(0, 1000);
}

function buildRetryDelayMs(attempt) {
  const multiplier = Math.max(1, attempt);
  return env.emailQueueBaseBackoffSeconds * 1000 * multiplier;
}

function buildDedupeWindow(hours) {
  const windowMs = Math.max(1, hours) * 60 * 60 * 1000;
  return Math.floor(Date.now() / windowMs);
}

class EmailQueue {
  constructor(emailService, db = prisma) {
    this.emailService = emailService;
    this.db = db;
    this.processing = false;
    this.timer = null;
  }

  async enqueue(job, db = this.db) {
    try {
      const createdJob = await db.emailJob.create({
        data: {
          type: job.type,
          recipientEmail: job.payload.email,
          payload: job.payload,
          tokenId: job.tokenId || undefined,
          userId: job.userId || undefined,
          dedupeKey: job.dedupeKey || undefined,
          maxAttempts: job.maxAttempts || env.emailQueueMaxAttempts,
          availableAt: job.availableAt || new Date(),
        },
      });

      return { created: true, job: createdJob };
    } catch (error) {
      if (
        error instanceof Prisma.PrismaClientKnownRequestError &&
        error.code === "P2002" &&
        job.dedupeKey
      ) {
        return { created: false, reason: "duplicate-dedupe-key" };
      }

      throw error;
    }
  }

  start() {
    if (this.timer) {
      return;
    }

    this.timer = setInterval(() => {
      this.drain().catch((error) => {
        console.error("[email-queue] drain failed", { error });
      });
    }, env.emailQueuePollIntervalMs);

    this.timer.unref?.();
    setImmediate(() => {
      this.drain().catch((error) => {
        console.error("[email-queue] initial drain failed", { error });
      });
    });
  }

  stop() {
    if (!this.timer) {
      return;
    }

    clearInterval(this.timer);
    this.timer = null;
  }

  async claimNextJob() {
    const lockDeadline = new Date(
      Date.now() - env.emailQueueLockTimeoutSeconds * 1000
    );

    const candidate = await this.db.emailJob.findFirst({
      where: {
        status: EmailDeliveryStatus.PENDING,
        availableAt: {
          lte: new Date(),
        },
        OR: [{ lockedAt: null }, { lockedAt: { lt: lockDeadline } }],
      },
      orderBy: [{ availableAt: "asc" }, { createdAt: "asc" }],
    });

    if (!candidate) {
      return null;
    }

    const claimResult = await this.db.emailJob.updateMany({
      where: {
        id: candidate.id,
        status: EmailDeliveryStatus.PENDING,
        OR: [{ lockedAt: null }, { lockedAt: { lt: lockDeadline } }],
      },
      data: {
        status: EmailDeliveryStatus.PROCESSING,
        lockedAt: new Date(),
        attempts: {
          increment: 1,
        },
      },
    });

    if (claimResult.count !== 1) {
      return null;
    }

    return this.db.emailJob.findUnique({
      where: { id: candidate.id },
    });
  }

  async drain() {
    if (this.processing) {
      return;
    }

    this.processing = true;
    try {
      while (true) {
        const job = await this.claimNextJob();
        if (!job) {
          break;
        }

        await this.handle(job);
      }
    } finally {
      this.processing = false;
    }
  }

  async handle(job) {
    const providerName = this.emailService.provider;
    await markEmailTokenAttempt({
      tokenId: job.tokenId,
      providerName,
      db: this.db,
    });

    try {
      const delivery = await this.dispatch(job);
      await this.db.$transaction(async (tx) => {
        await tx.emailJob.update({
          where: { id: job.id },
          data: {
            status: EmailDeliveryStatus.SENT,
            lockedAt: null,
            sentAt: new Date(),
            lastError: null,
            providerName: delivery.providerName,
            providerMessageId: delivery.providerMessageId || undefined,
          },
        });

        await markEmailTokenSent({
          tokenId: job.tokenId,
          providerName: delivery.providerName,
          providerMessageId: delivery.providerMessageId,
          db: tx,
        });
      });

      console.info("[email-queue] email sent", {
        jobId: job.id,
        type: job.type,
        recipientEmail: job.recipientEmail,
        providerName: delivery.providerName,
        providerMessageId: delivery.providerMessageId,
      });
    } catch (error) {
      const errorMessage =
        truncateError(error.providerMessage || error.message);
      const attempts = job.attempts;
      const retryable =
        error.emailTransient === true &&
        attempts < (job.maxAttempts || env.emailQueueMaxAttempts);

      await this.db.$transaction(async (tx) => {
        await tx.emailJob.update({
          where: { id: job.id },
          data: retryable
            ? {
                status: EmailDeliveryStatus.PENDING,
                lockedAt: null,
                availableAt: new Date(Date.now() + buildRetryDelayMs(attempts)),
                lastError: errorMessage,
              }
            : {
                status: EmailDeliveryStatus.FAILED,
                lockedAt: null,
                lastError: errorMessage,
              },
        });

        if (retryable) {
          await markEmailTokenPendingRetry({
            tokenId: job.tokenId,
            errorMessage,
            db: tx,
          });
        } else {
          await markEmailTokenFailed({
            tokenId: job.tokenId,
            errorMessage,
            db: tx,
          });
        }
      });

      console.error("[email-queue] failed to send email", {
        jobId: job.id,
        type: job.type,
        recipientEmail: job.recipientEmail,
        attempts,
        retryable,
        providerName,
        providerStatusCode: error.providerStatusCode,
        providerBody: error.providerBody,
        error: errorMessage,
      });
    }
  }

  async dispatch(job) {
    switch (job.type) {
      case EmailJobType.VERIFY_EMAIL:
        return this.emailService.sendVerificationEmail(job.payload);
      case EmailJobType.NEW_LOGIN_ALERT:
        return this.emailService.sendNewLoginAlert(job.payload);
      case EmailJobType.PASSWORD_RESET:
        return this.emailService.sendPasswordResetEmail(job.payload);
      default:
        throw new Error(`Unknown email job type: ${job.type}`);
    }
  }
}

const emailQueue = new EmailQueue(new EmailService());

module.exports = {
  EmailQueue,
  emailQueue,
  EmailJobType,
  buildDedupeWindow,
};
