class DialogueManager:
    def __init__(self):
        self.last_object = None
        self.last_location = None
        self.last_decision = None
        self.last_adj = None

    def resolve(self, text):
        """
        Basic resolution layer.
        Later we can plug in spell correction here.
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

        if frame.get("adjectives"):
            self.last_adj = frame["adjectives"][0]

        self.last_decision = frame.get("decision")

    def generate_reply(self, frame):

        decision = frame.get("decision")
        obj = frame.get("object")
        loc = frame.get("location")
        adjs = frame.get("adjectives", [])

        # ===================================================
        # QUERY HANDLING (STATUS QUESTIONS)
        # ===================================================

        if decision == "QUERY":
            return {
                "agent": "system",
                "text": self.handle_query()
            }

        # ===================================================
        # PLAN HANDLING WITH SMART QUESTIONING
        # ===================================================

        if decision == "PLAN":

            # ---- NEW: adjective present but no object ----
            if adjs and not obj:
                adj = adjs[0]
                self.last_adj = adj

                return {
                    "agent": "manipulator",
                    "text": f"Which {adj} object should I pick up?"
                }

            # ---- object missing but location present ----
            if not obj and loc:
                return {
                    "agent": "manipulator",
                    "text": "What object should I pick up?"
                }

            # ---- location missing but object present ----
            if obj and not loc:
                self.last_object = obj

                return {
                    "agent": "mobile",
                    "text": f"Where should I deliver the {obj}?"
                }

            # ---- both missing ----
            if not obj and not loc:
                return {
                    "agent": "system",
                    "text": "I need both an object and a destination."
                }

            # ---- all information available ----
            return {
                "agent": "system",
                "text": "Executing your request."
            }

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
                return {
                    "agent": "system",
                    "text": f"I need more information about: {', '.join(missing)}"
                }

            return {
                "agent": "system",
                "text": "Your request is unclear. Please rephrase."
            }

        # ===================================================
        # REJECT HANDLING
        # ===================================================

        if decision == "REJECT":
            return {
                "agent": "system",
                "text": "Sorry, I cannot perform that request."
            }

        # ===================================================
        # FALLBACK
        # ===================================================

        return {
            "agent": "system",
            "text": "I did not understand that request."
        }

    # =======================================================
    # QUERY RESPONSE LOGIC (UNCHANGED CORE IDEA)
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
