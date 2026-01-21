#!/usr/bin/env python3
"""
测试 fix_img_context.py 的功能
"""

from fix_img_context import _process_json_obj, IMG_CONTEXT_TEXT


def test_single_image_url():
    """测试：单个 image_url"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "808/0.jpg"}},
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "这是一张图片"}]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert modified, "应该标记为已修改"
    
    content = result["messages"][0]["content"]
    assert len(content) == 2, f"应该有 2 个 item，实际: {content}"
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "<IMG_CONTEXT>\n", f"实际: {content[1]['text']}"
    print("✓ test_single_image_url PASSED")


def test_image_url_with_text():
    """测试：image_url 后面有 text"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "808/0.jpg"}},
                    {"type": "text", "text": "描述这张图片"}
                ]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert modified, "应该标记为已修改"
    
    content = result["messages"][0]["content"]
    # 期望: image_url, text(<IMG_CONTEXT>\n), text(描述这张图片)
    assert len(content) == 3, f"应该有 3 个 item，实际: {content}"
    assert content[0]["type"] == "image_url"
    assert content[1]["text"] == "<IMG_CONTEXT>\n"
    assert content[2]["text"] == "描述这张图片"
    print("✓ test_image_url_with_text PASSED")


def test_text_before_image_url():
    """测试：text 在 image_url 前面"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请看这张图片"},
                    {"type": "image_url", "image_url": {"url": "808/0.jpg"}}
                ]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert modified, "应该标记为已修改"
    
    content = result["messages"][0]["content"]
    # 期望: text(请看这张图片), image_url, text(<IMG_CONTEXT>\n)
    assert len(content) == 3, f"应该有 3 个 item，实际: {content}"
    assert content[0]["text"] == "请看这张图片"
    assert content[1]["type"] == "image_url"
    assert content[2]["text"] == "<IMG_CONTEXT>\n"
    print("✓ test_text_before_image_url PASSED")


def test_multiple_image_urls():
    """测试：多个 image_url"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "808/0.jpg"}},
                    {"type": "image_url", "image_url": {"url": "808/1.jpg"}},
                ]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert modified, "应该标记为已修改"
    
    content = result["messages"][0]["content"]
    # 期望: image_url, text(<IMG_CONTEXT>\n), image_url, text(<IMG_CONTEXT>\n)
    assert len(content) == 4, f"应该有 4 个 item，实际: {content}"
    assert content[0]["type"] == "image_url"
    assert content[1]["text"] == "<IMG_CONTEXT>\n"
    assert content[2]["type"] == "image_url"
    assert content[3]["text"] == "<IMG_CONTEXT>\n"
    print("✓ test_multiple_image_urls PASSED")


def test_no_image():
    """测试：没有 image_url，不应该修改"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi"}]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert not modified, "不应该标记为已修改"
    
    user_text = result["messages"][0]["content"][0]["text"]
    assert user_text == "Hello", f"不应该修改 text，实际: {user_text}"
    print("✓ test_no_image PASSED")


def test_multi_turn_with_images():
    """测试：多轮对话，每轮都有图片"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "1.jpg"}},
                    {"type": "text", "text": "第一张图片"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "好的"}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "2.jpg"}},
                    {"type": "image_url", "image_url": {"url": "3.jpg"}},
                    {"type": "text", "text": "第二三张图片"}
                ]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert modified, "应该标记为已修改"
    
    # 第一轮 user
    content1 = result["messages"][0]["content"]
    assert len(content1) == 3, f"第一轮应该有 3 个 item，实际: {content1}"
    assert content1[0]["type"] == "image_url"
    assert content1[1]["text"] == "<IMG_CONTEXT>\n"
    assert content1[2]["text"] == "第一张图片"
    
    # 第二轮 user
    content2 = result["messages"][2]["content"]
    assert len(content2) == 5, f"第二轮应该有 5 个 item，实际: {content2}"
    assert content2[0]["type"] == "image_url"
    assert content2[1]["text"] == "<IMG_CONTEXT>\n"
    assert content2[2]["type"] == "image_url"
    assert content2[3]["text"] == "<IMG_CONTEXT>\n"
    assert content2[4]["text"] == "第二三张图片"
    print("✓ test_multi_turn_with_images PASSED")


def test_assistant_not_modified():
    """测试：assistant message 不应该被处理"""
    data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image_url", "image_url": {"url": "test.jpg"}},
                    {"type": "text", "text": "这是图片"}
                ]
            }
        ]
    }
    
    result, modified = _process_json_obj(data)
    assert not modified, "assistant 中的 image_url 不应该被处理"
    
    # assistant 的 content 不应该被修改
    assert len(result["messages"][1]["content"]) == 2
    print("✓ test_assistant_not_modified PASSED")


def run_all_tests():
    print("=" * 50)
    print("开始测试 fix_img_context.py")
    print("=" * 50)
    
    test_single_image_url()
    test_image_url_with_text()
    test_text_before_image_url()
    test_multiple_image_urls()
    test_no_image()
    test_multi_turn_with_images()
    test_assistant_not_modified()
    
    print("=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
