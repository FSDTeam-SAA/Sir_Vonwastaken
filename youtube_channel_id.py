import os
from googleapiclient.discovery import build

API_KEY = "AIzaSyApPJWcwufdPRI6yOY3HHAC163eqRZV_3M"
youtube = build("youtube", "v3", developerKey=API_KEY)


def get_channel_id_from_handle(handle):
    # Ensure handle starts with @
    handle_name = handle if handle.startswith("@") else f"@{handle}"

    response = (
        youtube.channels().list(part="id", forHandle=handle_name).execute()
    )

    items = response.get("items", [])
    if items:
        return items[0]["id"]
    return None


# Example usage
handles = ["@FreeHighQualityDocumentaries","@travpedia","@naturesmomentstv","@bbcearth","@FreeDocumentaryNature","@NewTravelInsight","@World.TourYT","@UltimateNatureDocs","@DiscoverWildlifeEN"]
channel_ids = []

for h in handles:
    cid = get_channel_id_from_handle(h)
    if cid:
        channel_ids.append(cid)
        print(f"{h} -> {cid}")

# Formatted string ready for your .env file
print("\nYour .env variable:")
print(f"YOUTUBE_WATCH_CHANNELS={','.join(channel_ids)}")