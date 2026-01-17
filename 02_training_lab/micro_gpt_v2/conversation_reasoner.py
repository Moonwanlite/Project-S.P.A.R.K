class ConversationReasoner:

    def __init__(self):
        self.last_object = None
        self.last_location = None
        self.last_decision = None

    def resolve(self, text):
        """
        Basic normalization layer.
        Future hooks: spelling correction, coreference, etc.
        """
        return text.strip().lower()

    def update_state(self, frame):
        """
        Keep memory of previous context.
        """

        if frame.get("object"):
            self.last_object = frame["object"]

        if frame.get("location"):
            self.last_location = frame["location"]

        self.last_decision = frame.get("decision")

    def postprocess(self, frame):
        """
        Map intent labels into planning modes.
        This is the key link between intent SLM and planner.
        """

        decision = frame.get("decision")

        if decision == "PLAN_GRAB":
            frame["mode"] = "MANIPULATOR_ONLY"

        elif decision == "PLAN_NAVIGATE":
            frame["mode"] = "MOBILE_ONLY"

        elif decision == "PLAN_FETCH":
            frame["mode"] = "FULL_MISSION"

        return frame

    def generate_reply(self, frame):
        """
        Natural language response generator.
        """

        decision = frame.get("decision")
        obj = frame.get("object")
        loc = frame.get("location")

        if decision == "QUERY":
            return self.handle_query()

        if decision in ["PLAN_GRAB", "PLAN_NAVIGATE", "PLAN_FETCH"]:
            return "Executing your request."

        if decision == "CLARIFY":

            missing = []

            if not obj:
                missing.append("object")

            if not loc:
                missing.append("location")

            if missing:
                return f"I need more information about: {', '.join(missing)}"

            return "Your request is unclear."

        if decision == "REJECT":
            return "Sorry, I cannot perform that request."

        return "I did not understand that request."

    def handle_query(self):
        """
        Intelligent status response.
        """

        status_lines = []

        if self.last_decision:
            status_lines.append("System is active and ready.")
        else:
            status_lines.append("System is running.")

        if self.last_object:
            status_lines.append(f"Last object: {self.last_object}")

        if self.last_location:
            status_lines.append(f"Last location: {self.last_location}")

        if not self.last_object and not self.last_location:
            status_lines.append("No recent task information available.")

        return " ".join(status_lines)
