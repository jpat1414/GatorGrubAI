import utils
import main_model
import augmentation

epochs = 1

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(main_model.grub_model, utils.train_loader, utils.train_N, augmentation.apply_augmentation, main_model.optimizer, main_model.loss_function)
    utils.validate(main_model.grub_model, utils.valid_loader, utils.valid_N, main_model.loss_function)