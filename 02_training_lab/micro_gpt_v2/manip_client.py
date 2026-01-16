import socket
import json

HOST = "localhost"
PORT = 5050


def main():

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))

    # Identify as manipulator
    client.send("manipulator".encode())

    print("Manipulator terminal online...\n")

    while True:

        data = client.recv(4096).decode()

        if not data:
            break

        packet = json.loads(data)

        print("\n=== MANIPULATOR VIEW ===")

        print("SYSTEM:", packet["reply"])

        plan = packet["plan"]

        print("\nPLAN OVERVIEW:")
        print(plan["overview"])

        print("\nMESSAGES FOR MANIPULATOR:")

        for msg in plan["messages"]:
            if msg["agent"] in ["manipulator", "system"]:
                print(f"{msg['agent'].upper()}: {msg['text']}")


if __name__ == "__main__":
    main()
