from pyModbusTCP.server import ModbusServer, DataBank

server = ModbusServer("84.88.129.200", 502, no_block=True)
print("Start server...")
server.start()
print("Server is online")

print("Prem ENTER per introduir unes coordenades")


while True:

    x = float(input("Introdueix coordenada X: "))
    x_positive = x >= 0
    if(x_positive != True):
        x = x * -1
    y = float(input("Introdueix coordenada Y: "))
    y_positive = y >= 0
    if(y_positive != True):
        y = y * -1
    z = float(input("Introdueix coordenada Z: "))
    z_positive = z >= 0
    if(z_positive != True):
        z = z * -1
    vec_x = float(input("Introdueix coordenada de rotació en X: "))
    vec_x_positive = vec_x >= 0
    if(vec_x_positive != True):
        vec_x = vec_x * -1
    vec_y = float(input("Introdueix coordenada de rotació en Y: "))
    vec_y_positive = vec_y >= 0
    if(vec_y_positive != True):
        vec_y = vec_y * -1
    vec_z = float(input("Introdueix coordenada de rotació en Z: "))
    vec_z_positive = vec_z >= 0
    if(vec_z_positive != True):
        vec_z = vec_z * -1

    x = int(round(x, 3) * 100)
    y = int(round(y, 3) * 100)
    z = int(round(z, 3) * 100)
    vec_x = int(round(vec_x, 3) * 10000)
    vec_y = int(round(vec_y, 3) * 10000)
    vec_z = int(round(vec_z, 3) * 10000)


    server.data_bank.set_holding_registers(0, [x_positive])
    server.data_bank.set_holding_registers(10, [y_positive])
    server.data_bank.set_holding_registers(20, [z_positive])
    server.data_bank.set_holding_registers(30, [vec_x_positive])
    server.data_bank.set_holding_registers(40, [vec_y_positive])
    server.data_bank.set_holding_registers(50, [vec_z_positive])
    server.data_bank.set_holding_registers(60, [x])
    server.data_bank.set_holding_registers(70, [y])
    server.data_bank.set_holding_registers(80, [z])
    server.data_bank.set_holding_registers(90, [vec_x])
    server.data_bank.set_holding_registers(100, [vec_y])
    server.data_bank.set_holding_registers(110, [vec_z])

    print("Coordenades X, Y, Z i rotació en X, Y, Z del nas enviades per Modbus: " + str(round(x, 3)) + ", " + str(round(y, 3)) + ", " + str(round(z, 3))
          + ", " + str(round(vec_x, 3)) + ", " + str(round(vec_y, 3)) + ", " + str(round(vec_z, 3)))

    print("Vols introduir noves coordenades? (s/n)")
    if(input() != 's'):
        break