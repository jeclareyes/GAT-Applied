# export/report_logger.py
import logging
from configs.settings import LOG_FILE

class ReportLogger:
    """
    Registra incidencias durante el flujo y exporta un .txt estructurado.
    """
    def __init__(self, logfile=None):
        self.logfile = logfile or LOG_FILE
        self.entries = []

    def log(self, level, category, year=None, element_id=None, coords=None,
            description='', tolerance=None):
        """Agrega una entrada de log.
        level: 'ERROR','WARNING','INFO'
        """
        entry = {
            'level': level,
            'category': category,
            'year': year,
            'element_id': element_id,
            'coords': coords,
            'description': description,
            'tolerance': tolerance
        }
        self.entries.append(entry)
        # También envía al logger estándar
        log_msg = f"{category} | Año: {year} | ELEMENT_ID: {element_id} | Coords: {coords} | {description}"
        if level == 'ERROR':
            logging.error(log_msg)
        elif level == 'WARNING':
            logging.warning(log_msg)
        else:
            logging.info(log_msg)

    def export(self):
        """Escribe todas las entradas en el archivo txt"""
        try:
            with open(self.logfile, 'w', encoding='utf-8') as f:
                for e in self.entries:
                    line = (
                        f"{e['level']} | {e['category']} | "
                        f"Año: {e['year']} | ELEMENT_ID: {e['element_id']} | "
                        f"Coords: {e['coords']} | "
                        f"Tolerancia: {e['tolerance']} | {e['description']}\n"
                    )
                    f.write(line)
            logging.info(f"Reporte de errores exportado a {self.logfile}")
        except Exception as ex:
            logging.error(f"Error exportando reporte de errores: {ex}")
