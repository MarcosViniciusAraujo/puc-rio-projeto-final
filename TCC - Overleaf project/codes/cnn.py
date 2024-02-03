def create_and_fit(learning_rate, epochs,  batch_size, x_train, y_train, x_val, y_val):
    
    # defining model
    
    model = Sequential()
    model.add(
        Conv2D(
            32, 
            (3,3), 
            strides=1, 
            padding='same', 
            activation='relu', 
            input_shape = (150,150,1)
        )
    )

    model.add(BatchNormalization()) # normaliza a camada anterior, acelera o treinamento
    model.add(
        MaxPool2D(
            (2,2), 
            strides=2, 
            padding='same'
            )
    )

    model.add(
        Conv2D(
            64, 
            (3,3), 
            strides=1, 
            padding='same', 
            activation='relu'
        )
    )

    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(
        MaxPool2D(
            (2,2), 
            strides=2, 
            padding='same'
        )
    )
    
    model.add(
        Conv2D(
            256, 
            (3,3), 
            strides=1, 
            padding='same', 
            activation='relu', 
            name='l_05'
        )
    )

    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))    
    model.add(Flatten()) # lineariza a imagem
    model.add(Dense(units = 128 , activation = 'relu'))
    
    model.add(Dropout(0.2))
    model.add(Dense(units = 1 , activation = 'sigmoid'))
    
    model.compile(optimizer = otimizador, 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy']
    )
    
    # fiting the model 
    history = model.fit(
        datagen.flow(x_train,y_train, batch_size=batch_size),
        epochs=epochs, 
        validation_data=datagen.flow(x_val, y_val),
        callbacks=[learning_rate])
    
    return history, model