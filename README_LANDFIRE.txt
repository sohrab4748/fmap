FMAP LANDFIRE Patch

This patch adds optional Google Earth Engine (LANDFIRE EVT/EVC/EVH) summaries to the FMAP run output
and adds an endpoint to find candidate "fireland" points within a drawn AOI.

Files:
- main.py: adds LANDFIRE summaries (analysis['landfire']) and POST /fmap/fireland_candidates
- index.html: adds LANDFIRE charts + "Find fireland points" button (Region mode)
- requirements.txt: adds earthengine-api

Render environment variables (required to enable LANDFIRE/Fireland):
- FMAP_ENABLE_EE=1
- EE_SERVICE_ACCOUNT=<service account email>
- EE_PRIVATE_KEY_JSON=<service account key JSON>   (OR EE_PRIVATE_KEY_JSON_B64=<base64 JSON>)
Optional:
- FMAP_LANDFIRE_EVT_CSV=/opt/render/project/src/landfire_evt_lookup.csv   (maps EVT code->name)

If Earth Engine isn't configured, LANDFIRE charts will show "not available" and Fireland search will error with a reason.
