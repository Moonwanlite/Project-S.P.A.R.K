import socket
import json

from run_jarvis import interpret, load_components
from planner import generate_plan
from dialogue_manager import DialogueManager


HOST = "localhost"
PORT = 5050


def main():

    print("Starting Brain Server...")

    tokenizer, intent_model, tag_model, intent_meta, tag_meta, id_to_decision, id_to_label = load_components()

    dialogue = DialogueManager()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(2)

    clients = {}

    print("Waiting for agents to connect...")

    # Wait for both agents
    while len(clients) < 2:
        conn, addr = server.accept()
        role = conn.recv(1024).decode()

        clients[role] = conn
        print(f"{role.upper()} connected from {addr}")

    print("Both agents connected. System ready.\n")

    while True:

        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            break

        frame = interpret(
            user_input,
            tokenizer,
            intent_model,
            tag_model,
            intent_meta,
            tag_meta,
            id_to_decision,
            id_to_label
        )

        dialogue.update_state(frame)

        reply = dialogue.generate_reply(frame)

        print("System Reply:", reply)

        plan = generate_plan(frame)

        packet = json.dumps({
            "reply": reply,
            "plan": plan
        })

        # Broadcast to both agents
        for conn in clients.values():
            conn.send(packet.encode())

    print("Shutting down brain server.")


if __name__ == "__main__":
    main()
