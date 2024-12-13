from flask import Flask, jsonify, request

app = Flask(__name__)

# Dummy graph object with an invoke method
class Graph:
    def __init__(self):
        self.state = {"status": "initialized"}

    def invoke(self):
        # Simulate some processing
        self.state["status"] = "invoked"
        return self.state

graph = Graph()

@app.route('/invoke', methods=['POST'])
def invoke_graph():
    result = graph.invoke()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)