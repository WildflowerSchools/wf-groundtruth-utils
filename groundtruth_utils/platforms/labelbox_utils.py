# Labelbox uses both an internal form for geom representation as well as uses the geojson format
# This function is provided to convert to geojson as needed


def labelbox_geom_to_geojson(geom):
    if geom is None:
        return None

    if "point" in geom:
        return {
            "type": "Point",
            "coordinates": [
                geom['point']['x'],
                geom['point']['y']
            ]
        }

    elif "bbox" in geom:
        max_x = geom['bbox']['left'] + geom['bbox']['width']
        min_y = geom['bbox']['top']
        min_x = geom['bbox']['left']
        max_y = geom['bbox']['top'] + geom['bbox']['height']
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                    [min_x, min_y]
                ]]
            ]
        }
