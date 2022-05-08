# -*- coding: UTF-8 -*-
"""
@author:潘越
@file:main.py
@time:2022/04/04
"""

import warnings
import json

from flask import Flask, Response, request

from check_tools import check

app = Flask(__name__)


def build_success(content):
    result = {
        'success': True,
        'message': '',
        'content': content
    }
    return Response(json.dumps(result), mimetype='application/json')


def build_failure(message):
    result = {
        'success': False,
        'message': message,
        'content': {}
    }
    return Response(json.dumps(result), mimetype='application/json')


@app.route("/api/model/check", methods=['GET'])
def api_model_check():
    try:
        text = request.args.get("text")
        crime = request.args.get("crime")

        text_list = check(text, crime)
        return build_success(text_list)
    except Exception as e:
        print('错误明细是：', e.__class__.__name__, e)
        return build_failure('检验失败')


if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0', threaded=True)
    warnings.filterwarnings('ignore')
