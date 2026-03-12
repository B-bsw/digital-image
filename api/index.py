from app.__main__ import app

# Vercel serverless handler
def handler(request, response):
    return app
