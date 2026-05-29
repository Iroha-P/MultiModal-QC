def test_parse_row_pass():
    from data.scripts.convert_drilling import parse_row

    row = {
        "well_id": "55773141",
        "design_depth": "12-15m",
        "actual_depth": 12.3,
        "measured_depth": 12.3,
        "result": "合格",
        "remark": "",
        "no_video": "",
        "similar": "",
        "blurry": "",
        "no_mark": "",
        "incomplete": "",
        "unreported": "",
        "depth_fail": "",
        "helmet": "",
        "uniform": "",
        "gloves": "",
    }

    qa = parse_row(row, "video_001.mp4")

    assert "合格" in qa["conversations"][1]["content"]
    assert qa["video"] == "video_001.mp4"


def test_parse_row_fail():
    from data.scripts.convert_drilling import parse_row

    row = {
        "well_id": "55962791",
        "design_depth": "12-15m",
        "actual_depth": 12.6,
        "measured_depth": None,
        "result": "不合格",
        "remark": "视频模糊看不清",
        "no_video": "",
        "similar": "",
        "blurry": "是",
        "no_mark": "",
        "incomplete": "",
        "unreported": "",
        "depth_fail": "",
        "helmet": "",
        "uniform": "",
        "gloves": "",
    }

    qa = parse_row(row, "video_042.mp4")
    content = qa["conversations"][1]["content"]

    assert "不合格" in content
    assert "模糊" in content
