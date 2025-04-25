#!/usr/bin/env python3
"""
main.py: Punto de entrada principal para el pipeline GIS.
Carga configuración y dispara la ejecución del pipeline completo.
"""
import sys
import argparse
from configs.logging_config import setup_logging

# Inicializar logging
setup_logging()

# Importar función principal del pipeline
from scripts.run_full_pipeline import main as run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Ejecuta el pipeline completo de consolidación y análisis de red vial GIS'
    )
    parser.add_argument(
        '--input-dir',
        help='Directorio con GeoPackages anuales (por defecto el definido en settings)',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        help='Directorio de salida para archivos generados (por defecto el definido en settings)',
        default=None
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        help='Tolerancia de matching geométrico en metros (por defecto el definido en settings)',
        default=None
    )
    args, extra = parser.parse_known_args()

    # Ajustar sys.argv para pasárselo al script interno
    # El script espera solo sus propios args
    new_argv = [sys.argv[0]]
    if args.input_dir:
        new_argv += ['--input-dir', args.input_dir]
    if args.output_dir:
        new_argv += ['--output-dir', args.output_dir]
    if args.tolerance is not None:
        new_argv += ['--tolerance', str(args.tolerance)]
    sys.argv = new_argv + extra

    # Ejecutar pipeline completo
    run_full_pipeline()


if __name__ == '__main__':
    main()
