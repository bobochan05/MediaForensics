from __future__ import annotations

from flask import Blueprint, jsonify, request

from .service import AuthService, ServiceResult


def create_auth_blueprint(auth_service: AuthService) -> Blueprint:
    bp = Blueprint("auth_api", __name__, url_prefix="/api/auth")

    def _parse_json_payload() -> dict:
        payload = request.get_json(silent=True) or {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _response_from_result(result: ServiceResult):
        response = jsonify(result.payload)
        response.status_code = result.status_code
        if result.status_code == 429 and "retry_after" in result.payload:
            response.headers["Retry-After"] = str(result.payload["retry_after"])
        if result.token_bundle is not None:
            auth_service.set_auth_cookies(response, result.token_bundle)
        return response

    @bp.get("/session")
    def session_status():
        return jsonify(auth_service.build_session_payload(request))

    @bp.post("/signup")
    def signup():
        result = auth_service.signup(_parse_json_payload(), request)
        return _response_from_result(result)

    @bp.post("/login")
    def login():
        result = auth_service.login(_parse_json_payload(), request)
        return _response_from_result(result)

    @bp.post("/google")
    def google_login():
        result = auth_service.google_login(_parse_json_payload(), request)
        return _response_from_result(result)

    @bp.post("/guest")
    def guest_login():
        result = auth_service.guest_login(request)
        return _response_from_result(result)

    @bp.post("/refresh")
    def refresh():
        result = auth_service.refresh(request)
        return _response_from_result(result)

    @bp.post("/logout")
    def logout():
        result = auth_service.logout(request)
        response = _response_from_result(result)
        auth_service.clear_auth_cookies(response)
        return response

    @bp.get("/me")
    def me():
        principal, error = auth_service.authenticate_request(request)
        if principal is None:
            return jsonify({"error": error["message"], "code": error["code"]}), 401
        return jsonify(
            {
                "status": "ok",
                "auth_state": principal.get("auth_mode"),
                "user_id": principal.get("user_id"),
                "user_email": principal.get("email"),
                "user_username": principal.get("username"),
                "guest_id": principal.get("guest_id"),
            }
        )

    return bp
