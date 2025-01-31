from flask import Flask, request, jsonify
import uuid
import re
import requests
# Import your chatbot logic
from chat_logic import ecommerce_graph  # Replace with your module name

app = Flask(__name__)
_printed = set()



# @app.route('/chat', methods=['POST'])
# def chat():
#     """Handle user messages received via smee.io webhook and return bot responses."""
#     try:
#         # Get data from the incoming request
#         data = request.json
#         print("Received Data:", data)

#         if not data or "rawdata" not in data:
#             return jsonify({"error": "Invalid payload format."}), 400

#         rawdata = data["rawdata"]
#         user_message = rawdata.get("message", {}).get("conversation", "")
#         receiver_jid = rawdata.get("key", {}).get("remoteJid", "")

#         if not user_message or not receiver_jid:
#             return jsonify({"error": "Message or receiver number missing."}), 400

#         # Extract only the numeric phone number
#         receiver_number = re.sub("[^0-9]", "", receiver_jid)

#         # Configuration and state
#         config = {
#             "configurable": {
#                 "customer_id": "CUST-001",
#                 "thread_id": str(uuid.uuid4()),
#             }
#         }

#         # Process message through ecommerce_graph
#         events = ecommerce_graph.stream(
#             {"messages": [("user", user_message)]}, config, stream_mode="values"
#         )
        
#         response = ""
#         for event in events:
#             if "messages" in event:
#                 for msg in event["messages"]:
#                     if hasattr(msg, "content") and msg.content and msg.id not in _printed:
#                         response = msg.content
#                         print("Bot Response:", response)
#                         _printed.add(msg.id)
#                         break
        
#         return jsonify({"answer": response, "receiver": receiver_number})  # Include receiver number in response
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle user messages received via smee.io webhook and return bot responses."""
    try:
        # Get data from the incoming request
        data = request.json
        print("Received Data:", data)

        if not data or "rawdata" not in data:
            return jsonify({"error": "Invalid payload format."}), 400

        rawdata = data["rawdata"]
        user_message = rawdata.get("message", {}).get("conversation", "")
        receiver_jid = rawdata.get("key", {}).get("remoteJid", "")

        if not user_message or not receiver_jid:
            return jsonify({"error": "Message or receiver number missing."}), 400

        # Extract only the numeric phone number
        receiver_number = re.sub("[^0-9]", "", receiver_jid)

        # Configuration and state
        config = {
            "configurable": {
                "customer_id": "CUST-001",
                "thread_id": str(uuid.uuid4()),
            }
        }

        # Process message through ecommerce_graph
        events = ecommerce_graph.stream(
            {"messages": [("user", user_message)]}, config, stream_mode="values"
        )
        
        response = ""
        for event in events:
            if "messages" in event:
                for msg in event["messages"]:
                    if hasattr(msg, "content") and msg.content and msg.id not in _printed:
                        response = msg.content
                        print("Bot Response:", response)
                        _printed.add(msg.id)
                        break
        
        # Forward response to /send endpoint
        forward_data = {"answer": response, "receiver": receiver_number}
        send_response = requests.post("http://localhost:5000/send", json=forward_data)
        print("Forwarded to /send:", send_response.json())

        return jsonify(forward_data)  # Return the original response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send', methods=['POST'])
def send():
    """Handles sending messages to users."""
    try:
        data = request.json
        print("Forwarded Message Data:", data)

        receiver = data.get("receiver")
        message = data.get("answer")

        if not receiver or not message:
            return jsonify({"error": "Receiver or message missing."}), 400

        # Simulate sending the message (replace with actual logic)
        print(f"Sending message: '{message}' to {receiver}")

        nodeurl = 'https://server.msgbucket.com/send'
        data = {
            'receiver': receiver,
            'msgtext': message,
            'token': 'iulgFnHzPg3KF2rma9I3',
        }

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        response = requests.post(nodeurl, data=data, headers=headers, timeout=30, verify=False)

        print(response.text)

        return jsonify({"status": "success", "sent_to": receiver, "message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)