import React from 'react';
import { Bar } from 'react-chartjs-2';

export default function VerticalBar(props) {
    const data = {
        labels: ['BERTCRF', 'BiLSTMCRF', 'BERT', 'CRF', 'BiLSTM'],
        datasets: [
            {
                label: 'Entities',
                backgroundColor: 'rgba(255,99,132,0.2)',
                borderColor: 'rgba(255,99,132,1)',
                borderWidth: 1,
                hoverBackgroundColor: 'rgba(255,99,132,0.4)',
                hoverBorderColor: 'rgba(255,99,132,1)',
                data: props.countList //[12, 13, 10, 14, 12, 15]
            }
        ]
    };

    const options = {
        scales: {
          yAxes: [
            {
              ticks: {
                beginAtZero: true,
              },
            },
          ],
        },
      }

    return (
        <div className="VerticalBar">
            <h2>Predicted Entities Graph</h2>
            <Bar
                data={data}
                width={50}
                height={20}
                options={options}
            />
        </div>
    );
};