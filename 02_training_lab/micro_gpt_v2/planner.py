def generate_plan(frame):

    decision = frame.get("decision")
    mode = frame.get("mode")

    obj = frame.get("object")
    loc = frame.get("location")

    # --------------------------------------------
    # Clarification / query handling
    # --------------------------------------------

    if decision == "CLARIFY":
        return {
            "overview": "SYSTEM:CLARIFY|NULL",
            "messages": [
                {"agent": "system", "text": "Need more information to proceed."}
            ]
        }

    if decision == "QUERY":
        return {
            "overview": "SYSTEM:QUERY|NULL",
            "messages": [
                {"agent": "system", "text": "Status query received."}
            ]
        }

    if decision == "REJECT":
        return {
            "overview": "SYSTEM:REJECT|NULL",
            "messages": [
                {"agent": "system", "text": "Request rejected as invalid."}
            ]
        }

    # --------------------------------------------
    # Verb-aware planning
    # --------------------------------------------

    if decision == "PLAN_GRAB":

        if not obj:
            return {
                "overview": "SYSTEM:CANNOT_PLAN_MISSING_INFO|NULL",
                "messages": [
                    {"agent": "system", "text": "Cannot grasp without object."}
                ]
            }

        return {
            "overview": f"MANIPULATOR:GRASP|{obj.upper()}",
            "messages": [
                {"agent": "manipulator", "text": f"Attempting to grasp {obj}"}
            ]
        }

    if decision == "PLAN_NAVIGATE":

        if not loc:
            return {
                "overview": "SYSTEM:CANNOT_PLAN_MISSING_INFO|NULL",
                "messages": [
                    {"agent": "system", "text": "Cannot navigate without location."}
                ]
            }

        return {
            "overview": f"MOBILE:NAVIGATE|{loc.upper()}",
            "messages": [
                {"agent": "mobile", "text": f"Navigating to {loc}"}
            ]
        }

    if decision == "PLAN_FETCH":

        if not obj or not loc:
            return {
                "overview": "SYSTEM:CANNOT_PLAN_MISSING_INFO|NULL",
                "messages": [
                    {"agent": "system", "text": "Fetch requires both object and location."}
                ]
            }

        plan_steps = [
            f"MANIPULATOR:SEARCH|{obj.upper()}",
            "MOBILE:NAVIGATE|MANIPULATOR_STATION",
            f"MANIPULATOR:GRASP|{obj.upper()}",
            "MOBILE:WAIT_FOR_LOAD|NULL",
            "MANIPULATOR:PLACE_ON_ROBOT|NULL",
            "SYNC:HANDOFF_INITIATED|NULL",
            "MANIPULATOR:RELEASE|NULL",
            f"MOBILE:NAVIGATE|{loc.upper()}",
            "SYSTEM:MISSION_COMPLETE|NULL"
        ]

        return {
            "overview": " -> ".join(plan_steps),
            "messages": [
                {"agent": "manipulator", "text": f"Searching for {obj}"},
                {"agent": "mobile", "text": "Moving to manipulator station"},
                {"agent": "manipulator", "text": f"Grasping {obj}"},
                {"agent": "mobile", "text": f"Delivering {obj} to {loc}"},
                {"agent": "system", "text": "Mission completed successfully"}
            ]
        }

    # --------------------------------------------
    # Fallback
    # --------------------------------------------

    return {
        "overview": "SYSTEM:UNKNOWN|NULL",
        "messages": [
            {"agent": "system", "text": "Unable to plan for this request."}
        ]
    }
