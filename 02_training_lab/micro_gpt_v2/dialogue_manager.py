class DialogueManager:
    def __init__(self):
        self.last_object = None
        self.last_location = None
        self.last_decision = None

    def resolve(self, text):
        """
        Basic resolution layer.
        Here we could add coreference or spelling correction later.
        For now we simply pass through.
        """
        return text.strip().lower()

    def update_state(self, frame):
        """
        Remember context from previous commands.
        """

        if frame.get("object"):
            self.last_object = frame["object"]

        if frame.get("location"):
            self.last_location = frame["location"]

        self.last_decision = frame.get("decision")

    def generate_reply(self, frame):
        decision = frame.get("decision")
        obj = frame.get("object")
        loc = frame.get("location")

        # ===================================================
        # QUERY HANDLING (NEW FEATURE)
        # ===================================================

        if decision == "QUERY":
            return self.handle_query()

        # ===================================================
        # PLAN HANDLING
        # ===================================================

        if decision == "PLAN":
            return "Executing your request."

        # ===================================================
        # CLARIFICATION LOGIC
        # ===================================================

        if decision == "CLARIFY":

            missing = []

            if not obj:
                missing.append("object")

            if not loc:
                missing.append("location")

            if missing:
                return f"I need more information about: {', '.join(missing)}"

            return "Your request is unclear. Please rephrase."

        # ===================================================
        # REJECT HANDLING
        # ===================================================

        if decision == "REJECT":
            return "Sorry, I cannot perform that request."

        # ===================================================
        # FALLBACK
        # ===================================================

        return "I did not understand that request."

    # =======================================================
    # NEW: QUERY RESPONSE LOGIC
    # =======================================================

    def handle_query(self):
        """
        Generate intelligent status replies based on stored context.
        """

        status_lines = []

        if self.last_decision == "PLAN":
            status_lines.append("The system is currently idle and awaiting new instructions.")
        else:
            status_lines.append("System is active and ready for commands.")

        if self.last_object:
            status_lines.append(f"Last referenced object: {self.last_object}")

        if self.last_location:
            status_lines.append(f"Last referenced location: {self.last_location}")

        if not self.last_object and not self.last_location:
            status_lines.append("No recent task information available.")

        return " ".join(status_lines)
