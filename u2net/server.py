from flask import Flask, request, jsonify

import inference

app = Flask(__name__)

@app.route('/u2net', methods=['POST'])
def u2net():
    input_path = request.json.get('path')

    output_path = inference.get_single_saliency(net, input_path)

    response = {'path': output_path}
    return jsonify(response)

if __name__ == '__main__':
    net = inference.initialize_model()
    app.run(debug=True, host='0.0.0.0', port=21454)
