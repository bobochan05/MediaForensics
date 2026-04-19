const { escapeHtml } = require("./template-utils");

function passwordResetEmailTemplate({ fullName, resetUrl, expiresInMinutes }) {
  const name = escapeHtml(fullName || "there");
  const safeResetUrl = escapeHtml(resetUrl);

  return {
    subject: "Reset your Tracelyt password",
    html: `
      <div style="background:#f5f7fb;padding:32px 16px;font-family:Arial,Helvetica,sans-serif;color:#132238;">
        <div style="max-width:620px;margin:0 auto;background:#ffffff;border:1px solid #dbe4f0;border-radius:18px;overflow:hidden;">
          <div style="padding:28px 32px;border-bottom:1px solid #e8eef6;">
            <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#60758d;font-weight:700;">Tracelyt Security</div>
            <h1 style="margin:10px 0 0 0;font-size:26px;color:#101828;">Password reset request</h1>
          </div>
          <div style="padding:28px 32px;line-height:1.7;color:#344054;font-size:15px;">
            <p style="margin-top:0;">Hi ${name},</p>
            <p>We received a request to reset the password for your Tracelyt account.</p>
            <p style="margin:28px 0;">
              <a href="${safeResetUrl}" style="display:inline-block;padding:14px 22px;border-radius:12px;background:#1d4ed8;color:#ffffff;text-decoration:none;font-weight:700;">Reset Password</a>
            </p>
            <p>This link expires in ${expiresInMinutes} minutes.</p>
            <p>If you did not request a password reset, you can ignore this message.</p>
          </div>
          <div style="padding:20px 32px;background:#f8fafc;border-top:1px solid #e8eef6;font-size:12px;color:#667085;">
            Tracelyt Intelligence System | Password recovery
          </div>
        </div>
      </div>
    `,
    text: [
      "Reset your Tracelyt password",
      "",
      `Hello ${fullName || "there"},`,
      `Reset your password here: ${resetUrl}`,
      `This link expires in ${expiresInMinutes} minutes.`,
    ].join("\n"),
  };
}

module.exports = { passwordResetEmailTemplate };
