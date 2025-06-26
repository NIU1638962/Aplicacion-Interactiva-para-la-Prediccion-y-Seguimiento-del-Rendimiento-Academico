# -*- coding: utf-8 -*- noqa
"""
Created on Sat Jun 21 00:05:47 2025

@author: JoelT
"""
import api
import configuration
import environment
import utils

if __name__ == '__main__':
    environment.init()
    configuration.reload_config()


app = environment.flask.Flask(__name__)
api_class = api.ModelApi(
    class_order_file_path=configuration.class_order_file_path,
    fields_file_path=configuration.fields_file_path,
    model_file_path=configuration.model_file_path,
    original_dataset_file_path=configuration.original_dataset_file_path,
)


@app.route('/')
def main():
    response = environment.flask.make_response(
        environment.flask.render_template('main.html'),
    )
    response.headers['Content-Type'] = 'text/html; charset=utf-8'

    return response


@app.route("/api-get-fields")
def api_get_fields():
    data = api_class.api_get_fields()

    response = environment.flask.make_response(environment.flask.jsonify(data))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/api-input-validation')
def api_input_validation():
    # Receive params from query string
    arguments = environment.flask.request.args.to_dict()
    parameters = utils.un_string_parameters(arguments)
    result = api_class.api_input_validation(**parameters)
    response = environment.flask.make_response(
        environment.flask.jsonify(result),
    )
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-predictor')
def api_predictor():
    # Receive params from query string
    arguments = environment.flask.request.args.to_dict()
    parameters = utils.un_string_parameters(arguments)
    result = api_class.api_predictor(**parameters)
    response = environment.flask.make_response(
        environment.flask.jsonify(result),
    )
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-recommendator')
def api_recommendator():
    # Receive params from query string
    arguments = environment.flask.request.args.to_dict()
    parameters = utils.un_string_parameters(arguments)
    result = api_class.api_recommendator(**parameters)
    response = environment.flask.make_response(
        environment.flask.jsonify(result),
    )
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-reload')
def api_reload():
    result = api_class.api_reload()
    response = environment.flask.make_response(
        environment.flask.jsonify(result),
    )
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/api-train-model')
def api_train_model():
    result = api_class.api_train_model()
    response = environment.flask.make_response(
        environment.flask.jsonify(result),
    )
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response


@app.route('/results')
def results():
    arguments = environment.flask.request.args.to_dict()

    response = environment.flask.make_response(
        environment.flask.render_template('results.html', params=arguments),
    )
    response.headers['Content-Type'] = 'text/html; charset=utf-8'

    return response


@app.route('/submit')
def submit():
    response = environment.flask.make_response(
        environment.flask.render_template('submit.html'),
    )
    response.headers['Content-Type'] = 'text/html; charset=utf-8'

    return response


@app.route('/test')
def test():
    response = environment.flask.make_response(
        environment.flask.render_template('test.html'),
    )
    response.headers['Content-Type'] = 'text/html; charset=utf-8'

    return response


if __name__ == '__main__':
    app.run(
        host=configuration.server_ip,
        port=configuration.server_port,
        debug=(environment.LOG_LEVEL == environment.logging.DEBUG),
    )
