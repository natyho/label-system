import xml.etree.ElementTree as ET
import os
# import uuid
#import logger

#logger = logger.GetLogger("xml_reader", "xml_element_tree").initial_logger()

class GetXml(object):
    doc = None
    root = None

    def __init__(self, path):
        self.doc = ET.parse(path)
        self.root = self.doc.getroot()

    def walk_labelme_data(self, xml_dom, dictionary):
        '''format:
        <annotation>
            <filename>DJI_0002_15FPSa.jpg</filename>
            <folder>Demo</folder>
            <source>...</source>
            <object>
                <name>crosswalk</name>
                <deleted>0</deleted>
                <verified>0</verified>
                <occluded>no</occluded>
                <attributes>...<attributes/>
                <parts>...</parts>
                <date>19-Feb-2019 05:18:21</date>
                <id>0</id>
                <polygon>
                    <username>albertlo</username>
                    <pt>
                        <x>81</x>
                        <y>389</y>
                    </pt>
                    ...
                </polygon>
            </object>
            <imagesize>
                <nrows>720</nrows>
                <ncols>1280</ncols>
            </imagesize>
        </annotation>        
        '''        
        if xml_dom.getchildren():
            if xml_dom.tag in ['object', 'pt']:
                obj = {}
                for child in xml_dom:
                    obj.update(self.walk_labelme_data(child, {}))
                    # print(obj[child.tag])
                if dictionary.get(xml_dom.tag):
                    dictionary[xml_dom.tag].append(obj)
                else:
                    dictionary.update({xml_dom.tag: [obj]})
            else:
                dictionary[xml_dom.tag] = {}
                for child in xml_dom:
                    # print(child.tag, end=' ')
                    dictionary[xml_dom.tag].update( self.walk_labelme_data(child, dictionary[xml_dom.tag]) )
        else:
            dictionary[xml_dom.tag] = xml_dom.text or ''
        return dictionary        

    def get_resolution(self):
        resolution = {}
        size = self.root.find("size")
        for child in size:
            resolution["{}".format(child.tag)] = child.text
        # logger.info(resolution)
        return resolution

    def get_field_info(self):
        return self.walk_labelme_data(self.root, {})['annotation']

    def resolution_transform(self, ratio_x, ratio_y):
        # print(self.root.findall("x"))
        for tag in self.root.iter("x"):
            tag.text = str(int(int(tag.text) * ratio_x))
        for tag in self.root.iter("y"):
            tag.text = str(int(int(tag.text) * ratio_x))
        return self.doc

if __name__ == '__main__':
    #resolution tranform for field file
    for root, dirs, files in os.walk("field"):
        for f in files:
            filepath = os.path.join(root, f)
            xml_reader = GetXml(filepath)
            xml_reader.resolution_transform(1280 / 1920, 720 / 1080)
            xml_reader.doc.write(filepath + ".new")
