const { escapeHtml } = require("./template-utils");

function newLoginAlertTemplate({ fullName, ipAddress, deviceLabel, timestamp }) {
  const name = escapeHtml(fullName || "there");
  const safeIpAddress = escapeHtml(ipAddress);
  const safeDeviceLabel = escapeHtml(deviceLabel);
  const safeTimestamp = escapeHtml(timestamp);

  return {
    subject: "New login detected on your Tracelyt account",
    html: `
      <div style="background:#f5f7fb;padding:32px 16px;font-family:Arial,Helvetica,sans-serif;color:#132238;">
        <div style="max-width:620px;margin:0 auto;background:#ffffff;border:1px solid #dbe4f0;border-radius:18px;overflow:hidden;">
          <div style="padding:28px 32px;border-bottom:1px solid #e8eef6;">
            <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#60758d;font-weight:700;">Tracelyt Security</div>
            <h1 style="margin:10px 0 0 0;font-size:26px;color:#101828;">New login alert</h1>
          </div>
          <div style="padding:28px 32px;line-height:1.7;color:#344054;font-size:15px;">
            <p style="margin-top:0;">Hi ${name},</p>
            <p>We noticed a login from a device or IP address that does not match recent activity on your Tracelyt account.</p>
            <div style="padding:16px 18px;border:1px solid #dbe4f0;border-radius:14px;background:#f8fafc;">
              <p style="margin:0 0 8px 0;"><strong>Time:</strong> ${safeTimestamp}</p>
              <p style="margin:0 0 8px 0;"><strong>IP address:</strong> ${safeIpAddress}</p>
              <p style="margin:0;"><strong>Device:</strong> ${safeDeviceLabel}</p>
            </div>
            <p style="margin-top:20px;">If this was you, no action is required. If you do not recognize this activity, rotate your password immediately and review active sessions.</p>
          </div>
          <div style="padding:20px 32px;background:#f8fafc;border-top:1px solid #e8eef6;font-size:12px;color:#667085;">
            Tracelyt Intelligence System | New device / IP login notification
          </div>
        </div>
      </div>
    `,
    text: [
      "New login detected on your Tracelyt account",
      "",
      `Hello ${fullName || "there"},`,
      "We noticed a login from a new device or IP address.",
      `Time: ${timestamp}`,
      `IP address: ${ipAddress}`,
      `Device: ${deviceLabel}`,
    ].join("\n"),
  };
}

module.exports = { newLoginAlertTemplate };
