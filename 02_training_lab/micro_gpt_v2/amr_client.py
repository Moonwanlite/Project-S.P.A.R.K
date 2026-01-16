import socket
import json

HOST = "localhost"
PORT = 5050


def main():

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))

    # Identify as mobile
    client.send("mobile".encode())

    print("Mobile bot terminal online...\n")

    while True:

        data = client.recv(4096).decode()

        if not data:
            break

        packet = json.loads(data)

        print("\n=== MOBILE BOT VIEW ===")

        print("SYSTEM:", packet["reply"])

        plan = packet["plan"]

        print("\nPLAN OVERVIEW:")
        print(plan["overview"])

        print("\nMESSAGES FOR MOBILE:")

        for msg in plan["messages"]:
            if msg["agent"] in ["mobile", "system"]:
                print(f"{msg['agent'].upper()}: {msg['text']}")


if __name__ == "__main__":
    main()
