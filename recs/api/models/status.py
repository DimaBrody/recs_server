def create_status(message=None, error=None, content=None):
    response = {"status": error if error else "OK"}
    if message:
        response["message"] = message

    if content:
        response["content"] = content

    # 64088 : 10, 241433 : 7, 19137: 8

    return response
