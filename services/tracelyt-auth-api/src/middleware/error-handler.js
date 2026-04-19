const { ZodError } = require("zod");
const { AppError } = require("../errors/app-error");

function errorHandler(err, _req, res, _next) {
  if (err instanceof ZodError) {
    return res.status(400).json({
      error: "Validation failed.",
      details: err.flatten(),
    });
  }

  if (err instanceof AppError) {
    return res.status(err.statusCode).json({
      error: err.message,
      details: err.details || null,
    });
  }

  console.error("[auth-api] unexpected error", err);
  return res.status(500).json({
    error: "Internal server error.",
  });
}

module.exports = { errorHandler };
