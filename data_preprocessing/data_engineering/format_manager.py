from abc import ABC, abstractmethod
import re

class FormatHandler(ABC):
    """
    Handle certain image file name and image id format
    """
    format_name = "AbstractFormat"

    @staticmethod
    @abstractmethod
    def can_handle_filename(filename):
        """Whether the handler can handle given filename"""
        pass
    
    @staticmethod
    @abstractmethod
    def extract_id_from_filename(filename):
        """Extract id from given filename"""
        pass

    @staticmethod
    @abstractmethod
    def can_handle_metadata_record(metadata_record):
        """Whether the handler can handle given metadata record"""
        pass
    
    @staticmethod
    @abstractmethod
    def expand_ids_from_metadata(metadata_record):
        """Expand id from record in the metadata"""
        pass

class TKRImgIdRangeHandler(FormatHandler):
    """Handle IMG_{id} filename & {id1}-{id2} metadata record"""
    format_name = "ImgRange"

    @staticmethod
    def can_handle_filename(filename):
        return filename.startswith("IMG_")
    
    @staticmethod
    def extract_id_from_filename(filename):
        match = re.fullmatch(r"IMG_(\d+)", filename)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def can_handle_metadata_record(metadata_record):
        return bool(re.fullmatch(r"\d+\s*-\s*\d+", metadata_record))
    
    @staticmethod
    def expand_ids_from_metadata(metadata_record):
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", metadata_record)
        if match:
            start_id = int(match.group(1))
            end_id = int(match.group(2))
            if start_id <= end_id:
                return [str(i) for i in range(start_id, end_id+1)]
        return []

class TKRLineChatIdsHandler(FormatHandler):
    """Handle line_oa_chat_date_{id} filename & line{id1}-{id2} metadata record"""
    format_name = "LineChatIds"

    @staticmethod
    def can_handle_filename(filename):
        return filename.startswith("line_oa_chat_")
    
    @staticmethod
    def extract_id_from_filename(filename):
        match = re.search(r"(\d+)$", filename)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def can_handle_metadata_record(metadata_record):
        return metadata_record.lower().startswith("line") and bool(re.search(r"\d+\s*-\s*\d+", metadata_record))

    @staticmethod
    def expand_ids_from_metadata(metadata_record):
        match = re.search(r"(\d+)\s*-\s*(\d+)", metadata_record)
        if match:
            id1 = str(match.group(1))
            id2 = str(match.group(2))
            return [id1, id2]
        return []

class TKRFormatManager():
    def __init__(self):
        self.handlers = []
        self.register_default_handlers()

    def register_handler(self, handler_class):
        """Resigter a new handler"""
        if handler_class not in self.handlers:
            self.handlers.append(handler_class)
            print(f"Registered handler: {handler_class.format_name}")
    
    def register_default_handlers(self):
        self.register_handler(TKRImgIdRangeHandler)
        self.register_handler(TKRLineChatIdsHandler)

    def extract_id(self, filename):
        """Extract id from given filename"""
        filename = str(filename)
        for handler in self.handlers:
            if handler.can_handle_filename(filename):
                return handler.extract_id_from_filename(filename)
        print(f"Warning: can't handle filename: {filename}.")
        return None
    
    def expand_ids(self, metadata_record):
        """Expand ids from given metadata record"""
        if not isinstance(metadata_record, str):
            return []
        
        # --- Handle empty string ---
        if not metadata_record.strip():
            return []
        
        for handler in self.handlers:
            if handler.can_handle_metadata_record(metadata_record):
                return handler.expand_ids_from_metadata(metadata_record)
        print(f"Warning: can't handle metadata record: {metadata_record}.")
        return [metadata_record]
