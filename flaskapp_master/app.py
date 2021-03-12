# -*- coding: utf-8 -*-
# @Time : 2021/3/4 21:10
# @Author : xiaojie
# @File : app.py
# @Software: PyCharm

from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/enter_face', methods=["GET"])
def enter_face():
    print("request.args: {}".format(request.args.to_dict()))
    print("request.form: {}".format(request.form.to_dict()))
    return render_template('enter_face.html')


@app.route('/recognize_face', methods=["GET"])
def recognize_face():
    print("request.args: {}".format(request.args.to_dict()))
    print("request.form: {}".format(request.form.to_dict()))
    return render_template('recognize_face.html')


if __name__ == "__main__":
    app.run(debug=False)
