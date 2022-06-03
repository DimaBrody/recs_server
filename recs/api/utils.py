import json


def get_body(request):
    body_unicode = request.body
    body = json.loads(body_unicode)

    return body
