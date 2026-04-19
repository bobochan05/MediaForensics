const crypto = require("crypto");
const UAParser = require("ua-parser-js");

function buildDeviceProfile(userAgent) {
  const parser = new UAParser(userAgent || "");
  const browser = parser.getBrowser();
  const os = parser.getOS();
  const device = parser.getDevice();

  const deviceLabelParts = [
    browser.name,
    browser.version,
    os.name,
    device.type || "desktop",
  ].filter(Boolean);

  const normalized = JSON.stringify({
    browserName: browser.name || "unknown",
    browserMajor: String(browser.version || "unknown").split(".")[0],
    osName: os.name || "unknown",
    deviceType: device.type || "desktop",
    vendor: device.vendor || "unknown",
  });

  return {
    fingerprint: crypto.createHash("sha256").update(normalized).digest("hex"),
    label: deviceLabelParts.join(" | ") || "Unknown device",
  };
}

module.exports = { buildDeviceProfile };
