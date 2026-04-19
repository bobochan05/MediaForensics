const { Resend } = require("resend");
const sendgridMail = require("@sendgrid/mail");
const { env } = require("../../config/env");
const { AppError } = require("../../errors/app-error");
const { verificationEmailTemplate } = require("./templates/verification-email");
const { newLoginAlertTemplate } = require("./templates/new-login-email");
const { passwordResetEmailTemplate } = require("./templates/password-reset-email");

function extractSendgridMessageId(response) {
  return (
    response?.headers?.["x-message-id"] ||
    response?.headers?.["X-Message-Id"] ||
    null
  );
}

function normalizeProviderError(providerName, error) {
  const statusCode =
    error?.code ||
    error?.response?.statusCode ||
    error?.statusCode ||
    500;
  const body = error?.response?.body || error?.response?.data || null;
  const message =
    error?.message ||
    body?.errors?.map((item) => item.message).join("; ") ||
    "Email delivery failed.";
  const transient =
    statusCode === 429 || (statusCode >= 500 && statusCode < 600);

  const wrapped = new AppError(502, `${providerName} email delivery failed.`);
  wrapped.providerName = providerName;
  wrapped.providerStatusCode = statusCode;
  wrapped.providerBody = body;
  wrapped.providerMessage = message;
  wrapped.emailTransient = transient;
  return wrapped;
}

class EmailService {
  constructor() {
    this.provider = env.emailProvider;
    this.from = env.emailFrom;

    if (!this.from.includes("@")) {
      throw new Error("EMAIL_FROM must contain a verified sender email address.");
    }

    if (this.provider === "resend") {
      this.resend = new Resend(env.resendApiKey);
    }

    if (this.provider === "sendgrid") {
      sendgridMail.setApiKey(env.sendgridApiKey);
    }
  }

  async sendEmail({ to, subject, html, text }) {
    if (!to) {
      throw new AppError(500, "Email recipient is required.");
    }

    try {
      if (this.provider === "resend") {
        const response = await this.resend.emails.send({
          from: this.from,
          to,
          subject,
          html,
          text,
        });

        return {
          providerName: "resend",
          providerMessageId: response?.data?.id || null,
          providerStatusCode: 202,
        };
      }

      if (this.provider === "sendgrid") {
        const [response] = await sendgridMail.send({
          from: this.from,
          to,
          subject,
          html,
          text,
        });

        return {
          providerName: "sendgrid",
          providerMessageId: extractSendgridMessageId(response),
          providerStatusCode: response?.statusCode || 202,
        };
      }
    } catch (error) {
      throw normalizeProviderError(this.provider, error);
    }

    throw new AppError(500, `Unsupported email provider: ${this.provider}`);
  }

  async sendVerificationEmail(payload) {
    return this.sendEmail({
      to: payload.email,
      ...verificationEmailTemplate(payload),
    });
  }

  async sendNewLoginAlert(payload) {
    return this.sendEmail({
      to: payload.email,
      ...newLoginAlertTemplate(payload),
    });
  }

  async sendPasswordResetEmail(payload) {
    return this.sendEmail({
      to: payload.email,
      ...passwordResetEmailTemplate(payload),
    });
  }
}

module.exports = { EmailService };
