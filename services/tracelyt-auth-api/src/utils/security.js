const crypto = require("crypto");
const bcrypt = require("bcrypt");

const BCRYPT_ROUNDS = 12;

async function hashPassword(password) {
  return bcrypt.hash(password, BCRYPT_ROUNDS);
}

async function verifyPassword(password, passwordHash) {
  return bcrypt.compare(password, passwordHash);
}

function createRandomToken(bytes = 32) {
  return crypto.randomBytes(bytes).toString("hex");
}

function hashToken(token) {
  return crypto.createHash("sha256").update(String(token)).digest("hex");
}

module.exports = {
  hashPassword,
  verifyPassword,
  createRandomToken,
  hashToken,
};
