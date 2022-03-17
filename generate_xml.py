import os
import xml.etree.ElementTree as ET

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

def dump_xml(img_info, obj_info, out_path):
        '''根据图片信息和目标信息写xml到指定路径

        Args:
            img_info: [list], [img_name, W, H, C]
            obj_info: [dict], {obj_name1: [[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax], ...], obj_name2: ...}
        '''

        assert out_path.split('.')[-1] == 'xml'
        out_dir, xml_name = os.path.split(out_path)
        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'folder')
        folder.text = out_dir
        filename = ET.SubElement(root, 'filename')
        img_ext = img_info[0].split('.')[-1]
        img_name = xml_name.replace('.xml', '.' + img_ext)
        filename.text = img_name
        path = ET.SubElement(root, 'path')
        path.text = os.path.join(out_dir, img_name)
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        width.text = str(img_info[1])
        height.text = str(img_info[2])
        depth.text = str(img_info[3])

        for obj_name, bbox in obj_info.items():
            for box in bbox:
                object_root = ET.SubElement(root, 'object')
                name = ET.SubElement(object_root, 'name')
                name.text = obj_name
                pose = ET.SubElement(object_root, 'pose')
                pose.text = "Unspecified"
                trunc = ET.SubElement(object_root, 'truncated')
                trunc.text = "0"
                diff = ET.SubElement(object_root, 'difficult')
                diff.text = "0"
                bndbox = ET.SubElement(object_root, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(box[0]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(box[1]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(box[2]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(box[3]))
        indent(root)
        tree = ET.ElementTree(root)
        tree.write(out_path)

def parse_xml_info(xml_path):
    ''' 解析xml文件信息

    解析出的xml信息包含2类：
    第一类是图像信息：图像名图像宽高,通道数
    第二类是包含的目标信息：目标类别和每类目标所有bbx的位置

    Args:
        xml_path:xml文件路径

    Return
        img_info: [list], [img_name, W, H, C]
        obj_info: [dict], {obj_name1: [[xmin,ymin,xmax,ymax], [xmin,ymin,xmax,ymax], ...], obj_name2: ...}
    '''
    assert os.path.exists(xml_path), "{0} does not exist!".format(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_name = root.find('filename').text
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    img_depth = int(root.find('size/depth').text)
    img_info = [img_name, img_width, img_height, img_depth]

    obj_info = {}
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        if obj_name not in obj_info.keys():
            obj_info[obj_name] = []
        obj_info[obj_name].append((xmin, ymin, xmax, ymax))

    return img_info, obj_info
