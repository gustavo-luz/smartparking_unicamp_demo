# Parking Availability Totem - Hardware Implementation

[← Back to Main README](../README.md)

todo: adicionar como montar do 0, foto do circuito, o que estava lá no artigo anteriormente, pode ser detalhista aqui.

## Circuit Overview
![Figure 4: Totem Circuit Diagram](../../assets/docs/totem_schematic.png)

The totem features two LED panels (front and back) displaying real-time parking availability. The system operates as follows:

1. **Microcontroller**: ESP8266 retrieves parking data from InfluxDB every minute
2. **Display**: Shows available spaces on two 4-digit 7-segment panels
3. **Power**: 12V input regulated to 5V for logic components

## Key Components
| Component | Specification | Qty |
|-----------|---------------|-----|
| Voltage Regulator | LM2596 (12V→5V) | 1 |
| Multiplexer | CD4511BE | 2 |
| MOSFET | TIP122 | 14 |
| LED Panel | 4-digit 7-segment (Common Anode) | 2 |

### Installation Guide
Wiring Instructions