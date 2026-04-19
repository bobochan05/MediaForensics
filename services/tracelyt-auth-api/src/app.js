const cors = require("cors");
const express = require("express");
const helmet = require("helmet");
const morgan = require("morgan");
const { env } = require("./config/env");
const { errorHandler } = require("./middleware/error-handler");
const { authRouter } = require("./routes/auth.routes");

const app = express();

app.set("trust proxy", 1);
app.use(helmet());
app.use(cors({ origin: env.corsOrigin, credentials: true }));
app.use(express.json({ limit: "1mb" }));
app.use(morgan(env.nodeEnv === "production" ? "combined" : "dev"));

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "tracelyt-auth-api",
    timestamp: new Date().toISOString(),
  });
});

app.use("/api/auth", authRouter);
app.use(errorHandler);

module.exports = { app };
