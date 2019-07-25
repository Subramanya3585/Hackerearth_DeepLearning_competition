
#model2
model_name = "LaNet5"
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',
                input_shape= (128,128,3)))
model.add(Conv2D(kernel_size=(3, 3), filters=6, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(5, 5), filters=16, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(5, 5), filters=120, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(Flatten())
model.add(Dense(output_dim=120, activation='relu'))
model.add(Dense(output_dim=120, activation='relu'))
model.add(Dense(output_dim=30, activation='softmax'))

adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


history = model.fit_generator(datagen.flow(Xtrain,Ytrain, batch_size=batch_size), epochs = epochs, validation_data = (Xvalid,Yvaild),verbose = 2, steps_per_epoch=Xtrain.shape[0] // batch_size  , callbacks=[learning_rate_reduction])
