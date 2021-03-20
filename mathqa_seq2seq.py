"""Seq2Seq with MathQA program"""

import logging

from mathqa_processing import MathQAManager
from simple_seq2seq import SimpleEncoderDecoder

import config
logging.basicConfig(level=logging.INFO)


# TODO: put out the training code
# TODO: convert to pytorch
# TODO: implement macro-extraction
# TODO: deal with bad programs
def main():
    # load the data
    dummy = True
    mathqa_manager = MathQAManager(root_dir=config.MATHQA_DIR, max_vocabulary_size=config.MAX_VOCABULARY_SIZE, dummy=dummy)

    # create the model
    model = SimpleEncoderDecoder(source_vocab_size=mathqa_manager.text_vocabulary_size,
                                 target_vocab_size=mathqa_manager.code_vocabulary_size,
                                 internal_dim=config.INTERNAL_DIM,
                                 end_of_sequence_token=mathqa_manager.return_token_index)

    # train
    n_epochs = 20
    batch_size = 32
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam()

    teacher_forcing = True
    for epoch_num in range(n_epochs):
        mean_loss = keras.metrics.Mean()
        correctness = keras.metrics.Mean()
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        for i, inputs in enumerate(mathqa_manager.get_dataset_iterator('train', batch_size, shuffle=True)):
            targets = inputs.code_vector
            _, target_seq_len = targets.shape
            mask = tf.not_equal(targets, 0)
            with tf.GradientTape() as tape:
                output = model((inputs.text_vector, inputs.code_vector), teacher_forcing=teacher_forcing)
                loss = loss_object(targets, output, sample_weight=mask)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            mean_loss.update_state(loss)
            accuracy.update_state(targets, output, sample_weight=mask)
            correctness.update_state(mathqa_manager.check_correctness(output, inputs.linear_formula,
                                                                      inputs.extracted_numbers))

        logging.info(f"epoch={epoch_num} loss={mean_loss.result()}, accuracy={accuracy.result()}, "
                     f"correctness={correctness.result()}")
        if teacher_forcing and accuracy.result() > 0.5:
            teacher_forcing = False
            logging.info(f"***Stopping teacher forcing at epoch={epoch_num}***")

    logging.info("training finished, saving model...")
    model_path = 'simple_model_checkpoint'
    model.save_weights(model_path)

    # evaluation
    teacher_forcing = False

    accuracy = keras.metrics.SparseCategoricalAccuracy()
    mean_loss = keras.metrics.Mean()
    correctness = keras.metrics.Mean()

    for i, inputs in enumerate(mathqa_manager.get_dataset_iterator('test', batch_size)):
        targets = inputs.code_vector
        _, target_seq_len = targets.shape
        mask = tf.not_equal(targets, 0)
        output = model((inputs.text_vector, inputs.code_vector), teacher_forcing=teacher_forcing)
        loss = loss_object(targets, output, sample_weight=mask)
        mean_loss.update_state(loss)
        accuracy.update_state(targets, output, sample_weight=mask)
        correctness.update_state(mathqa_manager.check_correctness(output, inputs.linear_formula,
                                                                  inputs.extracted_numbers))

    logging.info(f"test accuracy={accuracy.result()}, test loss={mean_loss.result()}, "
                 f"test correctness={correctness.result()}")


if __name__ == "__main__":
    main()
