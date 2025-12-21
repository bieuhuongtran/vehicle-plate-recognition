def recognizeLicensePlate(imgBuffer):
    """Creates and returns a dictionary with person details."""
    result = []

    # Processing img to plates
    result.append({
        "x": 0,
        "y": 0,
        "w": 100,
        "h": 100,
        "text": "51A12345",
    })

    return result