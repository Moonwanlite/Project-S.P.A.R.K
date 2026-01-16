def generate_plan(frame):
    """
    Enhanced planner with decentralized execution.

    - If both object and location exist → full coordinated plan
    - If only location exists → mobile bot acts alone
    - If only object exists → manipulator acts alone
    - If neither → clarification required
    """

    decision = frame.get("decision")
    obj = frame.get("object")
    loc = frame.get("location")

    messages = []

    # =====================================================
    # NON-PLAN DECISIONS
    # =====================================================

    if decision != "PLAN":
        return {
            "overview": f"SYSTEM:{decision}|NULL",
            "messages": [
                {
                    "agent": "system",
                    "text": f"No executable plan for decision type: {decision}"
                }
            ]
        }

    # =====================================================
    # CASE 1: ONLY LOCATION PROVIDED
    # Mobile bot can still act
    # =====================================================

    if loc and not obj:
        loc_token = loc.upper().replace(" ", "_")

        overview = f"MOBILE:NAVIGATE|{loc_token}"

        messages.append({
            "agent": "mobile",
            "text": f"Navigating to {loc}"
        })

        return {
            "overview": overview,
            "messages": messages
        }

    # =====================================================
    # CASE 2: ONLY OBJECT PROVIDED
    # Manipulator can still act
    # =====================================================

    if obj and not loc:
        obj_token = obj.upper().replace(" ", "_")

        overview = f"MANIPULATOR:GRASP|{obj_token}"

        messages.append({
            "agent": "manipulator",
            "text": f"Attempting to grasp {obj}"
        })

        return {
            "overview": overview,
            "messages": messages
        }

    # =====================================================
    # CASE 3: FULL INFORMATION PROVIDED
    # Coordinated mission
    # =====================================================

    if obj and loc:
        obj_token = obj.upper().replace(" ", "_")
        loc_token = loc.upper().replace(" ", "_")

        overview = " -> ".join([
            f"MANIPULATOR:SEARCH|{obj_token}",
            "MOBILE:NAVIGATE|MANIPULATOR_STATION",
            f"MANIPULATOR:GRASP|{obj_token}",
            "MOBILE:WAIT_FOR_LOAD|NULL",
            "MANIPULATOR:PLACE_ON_ROBOT|NULL",
            "SYNC:HANDOFF_INITIATED|NULL",
            "MANIPULATOR:RELEASE|NULL",
            f"MOBILE:NAVIGATE|{loc_token}",
            "SYSTEM:MISSION_COMPLETE|NULL"
        ])

        messages = [
            {
                "agent": "manipulator",
                "text": f"Searching for {obj}"
            },
            {
                "agent": "mobile",
                "text": "Moving to manipulator station"
            },
            {
                "agent": "manipulator",
                "text": f"Grasping {obj}"
            },
            {
                "agent": "mobile",
                "text": "Waiting for object to be loaded"
            },
            {
                "agent": "manipulator",
                "text": "Placing object on robot"
            },
            {
                "agent": "mobile",
                "text": f"Delivering {obj} to {loc}"
            },
            {
                "agent": "system",
                "text": "Mission completed successfully"
            }
        ]

        return {
            "overview": overview,
            "messages": messages
        }

    # =====================================================
    # CASE 4: NO INFORMATION AT ALL
    # =====================================================

    return {
        "overview": "SYSTEM:CANNOT_PLAN_MISSING_INFO|NULL",
        "messages": [
            {
                "agent": "system",
                "text": "Cannot generate plan due to missing object and location."
            }
        ]
    }