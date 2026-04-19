function getClientIp(req) {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.trim()) {
    return forwarded.split(",")[0].trim();
  }

  return (
    req.ip ||
    req.socket?.remoteAddress ||
    "unknown"
  );
}

function getUserAgent(req) {
  return String(req.headers["user-agent"] || "unknown").slice(0, 1024);
}

module.exports = {
  getClientIp,
  getUserAgent,
};
