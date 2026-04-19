const { app } = require("./app");
const { env } = require("./config/env");
const { prisma } = require("./lib/prisma");
const { emailQueue } = require("./services/email/email-queue");

async function start() {
  await prisma.$connect();
  emailQueue.start();

  const server = app.listen(env.port, () => {
    console.log(`[tracelyt-auth-api] listening on http://localhost:${env.port}`);
  });

  async function shutdown(signal) {
    console.log(`[tracelyt-auth-api] received ${signal}, shutting down`);
    emailQueue.stop();
    await prisma.$disconnect();
    server.close(() => process.exit(0));
  }

  process.on("SIGINT", () => {
    shutdown("SIGINT").catch((error) => {
      console.error("[tracelyt-auth-api] shutdown failed", error);
      process.exit(1);
    });
  });

  process.on("SIGTERM", () => {
    shutdown("SIGTERM").catch((error) => {
      console.error("[tracelyt-auth-api] shutdown failed", error);
      process.exit(1);
    });
  });
}

start().catch((error) => {
  console.error("[tracelyt-auth-api] failed to start", error);
  process.exit(1);
});
